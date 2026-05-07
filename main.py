import streamlit as st
import os
import re
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import cv2
import tempfile
import traceback
import html
import requests
from rapidfuzz import fuzz as rfuzz

# Attempt to import googletrans (for offline/no-key translation).
try:
    from googletrans import Translator
    _translator_available = True
    _translator = Translator()
except Exception:
    _translator_available = False
    _translator = None

# App UI configuration (professional blue)

st.set_page_config(page_title="🩺 SwasthyaSaar", page_icon="💉", layout="wide")
st.markdown(
    """
    <style>
      body { background: #f6fbff; }
      .main-title { font-size: 24px; color: #0b63b8; font-weight: 800; margin-bottom: 3px; }
      .subtitle { color: #2b2b2b; margin-top: 0px; margin-bottom: 12px; }
      .stButton>button { background-color: #0078D7; color: white; border-radius: 8px; }
      .stDownloadButton>button { background-color: #005fa3; color: white; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🩺 SwasthyaSaar —An NLP-Based Solution for Simplifying and Summarizing Medical Reports</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a lab report (PDF/image/txt).Medical Report Simplifier.</div>', unsafe_allow_html=True)

# Paths & config

POSSIBLE_DATA_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    r"C:\Users\Pooja\nlp project\data",
    "/mnt/data",
    "."
]
def find_file(fname):
    for d in POSSIBLE_DATA_DIRS:
        path = os.path.join(d, fname)
        if os.path.exists(path):
            return path
    return None

POPPLER_PATH = os.path.join(os.environ.get("LOCALAPPDATA", ""), "poppler", "poppler-24.08.0", "Library", "bin")  
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  
FINE_TUNED_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "lab_summarizer")  

# Load glossary, readme examples, reference ranges 

@st.cache_data
def load_glossary():
    fname = find_file("glossary - glossary.csv") or find_file("glossary.csv")
    glossary = {}
    if not fname:
        return glossary
    try:
        df = pd.read_csv(fname, engine="python")
        term_col, simple_col = None, None
        cols_lower = [c.lower().strip() for c in df.columns]
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ("term", "word", "token", "parameter", "param", "name"):
                term_col = c
            if cl in ("simple", "meaning", "definition", "gloss"):
                simple_col = c
        if term_col is None:
            term_col = df.columns[0]
        if simple_col is None and df.shape[1] > 1:
            simple_col = df.columns[1]
        for _, row in df.iterrows():
            key = str(row[term_col]).strip().lower()
            val = str(row[simple_col]).strip() if simple_col else ""
            if key:
                glossary[key] = val
                # Also add indexed keys: if term is "Hemoglobin (Hb or Hgb)", add "hb" and "hgb" as keys too
                # Extract acronyms from parentheses and add them
                paren_match = re.search(r'\(([^)]+)\)', key)
                if paren_match:
                    inner = paren_match.group(1).strip()
                    for token in re.split(r'\s+or\s+|\s+/\s+|,', inner):
                        token = token.strip().lower()
                        if token and len(token) > 0:
                            glossary[token] = val
    except Exception as e:
        st.warning(f"Failed to load glossary: {e}")
    return glossary

def load_reference_ranges():
    """Load reference ranges and build section-to-parameter mapping from # comments."""
    fname = find_file("reference_ranges.csv")
    reference = {}
    section_map = {}  # section_name -> [param_names]
    if not fname:
        return reference, section_map
    try:
        # First pass: read raw lines to build section mapping
        current_section = "general"
        param_sections = {}  # param -> section
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    current_section = line.lstrip("# ").strip().lower()
                    if current_section not in section_map:
                        section_map[current_section] = []
                elif line and not line.startswith("parameter"):
                    param_name = line.split(",")[0].strip().lower()
                    if param_name:
                        param_sections[param_name] = current_section
                        section_map.setdefault(current_section, []).append(param_name)

        # Second pass: load data with pandas (skip # lines)
        df = pd.read_csv(fname, engine="python", comment='#')
        param_col = None
        low_col = None
        high_col = None
        unit_col = None
        syn_col = None
        for c in df.columns:
            lc = c.lower().strip()
            if lc in ("parameter", "param", "name"):
                param_col = c
            if lc in ("low", "min", "lbound"):
                low_col = c
            if lc in ("high", "max", "ubound"):
                high_col = c
            if lc in ("unit", "units"):
                unit_col = c
            if lc in ("synonyms", "synonym"):
                syn_col = c
        if param_col is None:
            param_col = df.columns[0]

        def parse_bound(x):
            if pd.isna(x): return None
            s = str(x).strip()
            s = s.replace("%", "")
            s = re.sub(r"[^\d\.\-]", "", s)
            if not s: return None
            if "-" in s:
                parts = [p for p in s.split("-") if p]
                try:
                    nums = [float(p) for p in parts]
                    return sum(nums) / len(nums)
                except:
                    return None
            try:
                return float(s)
            except:
                return None

        for _, row in df.iterrows():
            p = str(row[param_col]).strip().lower()
            low = parse_bound(row[low_col]) if low_col else None
            high = parse_bound(row[high_col]) if high_col else None
            unit = ""
            syns = []
            if unit_col:
                unit = str(row[unit_col]).strip() if not pd.isna(row[unit_col]) else ""
            if syn_col:
                raw = str(row[syn_col]) if not pd.isna(row[syn_col]) else ""
                syns = [s.strip().lower() for s in re.split(r"[;,/|]", raw) if s.strip()]
            if p:
                reference[p] = {"low": low, "high": high, "unit": unit, "synonyms": syns, "section": param_sections.get(p, "general")}
    except Exception as e:
        st.error(f"Failed to load reference_ranges.csv: {e}")
    return reference, section_map

glossary_map = load_glossary()
reference_ranges, section_map = load_reference_ranges()

@st.cache_data
def load_medical_corpus():
    """Load medical_corpus.json for Hindi/Marathi meanings"""
    import json
    fname = find_file("medical_corpus.json")
    corpus = {}
    if not fname:
        return corpus
    try:
        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)
        for term, info in data.get("medical_terms", {}).items():
            corpus[term.lower()] = info
    except Exception:
        pass
    return corpus

medical_corpus = load_medical_corpus()

@st.cache_data
def load_medical_jargon():
    """Load medical_jargon.json — broad medical abbreviation/term definitions"""
    import json, ast
    fname = find_file("medical_jargon.json")
    if not fname:
        # Also check data/ directory
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "medical_jargon.json")
        if os.path.exists(data_path):
            fname = data_path
    jargon = {}
    if not fname:
        return jargon
    try:
        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)
        for term, raw_val in data.items():
            key = term.strip().lower()
            if not key:
                continue
            # Values are stored as string repr of lists, e.g. "['definition text']"
            try:
                parsed = ast.literal_eval(raw_val)
                if isinstance(parsed, list):
                    # Take the longest meaningful string from the list
                    meanings = [s.strip() for s in parsed if isinstance(s, str) and len(s.strip()) > 5]
                    if meanings:
                        jargon[key] = max(meanings, key=len)
                else:
                    jargon[key] = str(parsed).strip()
            except Exception:
                jargon[key] = str(raw_val).strip()
    except Exception:
        pass
    return jargon

medical_jargon = load_medical_jargon()


def _sanitize_meaning(text):
    """
    Clean a meaning string:
     - unescape HTML entities
     - remove newlines / excessive whitespace
     - keep first sentence only
     - reject clearly model-like, translation, or observation text
    """
    if not text:
        return ""
    s = html.unescape(str(text)).strip()
    s = re.sub(r"\s+", " ", s)
    
    # Keep only first sentence early
    if "." in s:
        s = s.split(".")[0].strip() + "."

    # Reject obvious LLM/translation/example artifacts
    if re.search(r"\bgpt\b|\bchatgpt\b|\bmodel\b|\btranslate\b", s, re.I):
        return ""

    # Reject observation/report-like fragments
    if re.search(r"(patient|report|shows?|example|sample|value(?:s)?|count at|observed|his|her|with\s+\w+\s+and)", s, re.I):
        return ""

    # Reject if it contains digits (likely a specific observation, not a definition)
    if re.search(r"\d", s):
        return ""

    # Reject very long strings unless they contain clear definition keywords
    if len(s) > 250 and not re.search(
        r"(measure|measures|average|concentration|protein|found in|amount of|transport|oxygen|blood|indicates|is an|refers to|type of)",
        s, re.I
    ):
        return ""

    return s


# -------------------------------
# Summarizer (fine-tuned fallback)
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_summarizer():
    """
    Loads the summarization pipeline once and caches it.
    Returns either:
    - fine-tuned model if present
    - huggingface bart-large-cnn as fallback
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    if os.path.exists(FINE_TUNED_MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(FINE_TUNED_MODEL_PATH)
        return pipeline("summarization", model=model, tokenizer=tokenizer)
    
    # fallback to Hugging Face BART
    return pipeline("summarization", model="facebook/bart-large-cnn")

# Load summarizer once at start
summarizer = get_summarizer()

def expand_abbreviations_for_summary(text, param_map):
    # param_map: param -> full_form (if available)
    for p, full in param_map.items():
        if full:
            # replace exact word occurrences (case-insensitive)
            text = re.sub(rf"\b{re.escape(p)}\b", f"{p.upper()} ({full})", text, flags=re.IGNORECASE)
    return text

#Generate AI summary
# -------------------------------
@st.cache_data
def simplify_meaning(param, raw_meaning):
    """Clean a medical definition into short plain English."""
    if not raw_meaning or len(raw_meaning.strip()) < 5:
        return ""
    
    # Take first meaningful clause only
    s = raw_meaning.split(";")[0].strip()
    # Remove param name echo (e.g., "hemoglobin is protein that..." → "protein that...")
    s = re.sub(rf"^{re.escape(param)}\s*(is|are|means?|[-–:])\s*", "", s, flags=re.I)
    # Remove leading articles
    s = re.sub(r"^(a |an |the )", "", s, flags=re.I)
    # Take only first comma-clause if still too long
    if len(s) > 60:
        s = s.split(",")[0].strip()
    if len(s) > 60:
        s = s[:60].rsplit(" ", 1)[0]
    
    # Now use BART to further simplify if the meaning is still complex
    if len(s) > 30 and any(w in s.lower() for w in ["concentration", "metabolism", "morphology", "pathology", "serum", "enzyme"]):
        try:
            prompt = f"Simplify for patient: {s}"
            out = summarizer(prompt, max_length=20, min_length=5, do_sample=False)
            simplified = out[0]["summary_text"].strip()
            # Only use if it's actually shorter and different
            if simplified and len(simplified) < len(s) and simplified.lower() != s.lower():
                s = simplified
        except Exception:
            pass
    
    return s.lower().rstrip(".")


def get_simple_explanation(param):
    """Get a simple explanation for a parameter."""
    meta = param_meta.get(param, {})
    raw_meaning = meta.get("meaning", "")
    if not raw_meaning:
        return ""
    return simplify_meaning(param, raw_meaning)


def _get_health_impact(param, status, raw_meaning):
    """Generate a health impact message based on param category."""
    # Categorize by keywords in meaning/param name
    param_lower = param.lower()
    meaning_lower = (raw_meaning or "").lower()
    combined = param_lower + " " + meaning_lower
    
    if status == "Low":
        if any(w in combined for w in ["oxygen", "hemoglobin", "rbc", "red blood", "hematocrit"]):
            return "you may feel tired, weak, or short of breath"
        elif any(w in combined for w in ["white blood", "wbc", "neutrophil", "lymphocyte", "immune"]):
            return "your body may have trouble fighting infections"
        elif any(w in combined for w in ["platelet", "clot", "bleeding"]):
            return "you may bruise or bleed more easily"
        elif any(w in combined for w in ["iron", "ferritin"]):
            return "you may feel exhausted or look pale"
        elif any(w in combined for w in ["vitamin d", "calcium", "bone"]):
            return "your bones may become weak over time"
        elif any(w in combined for w in ["vitamin b12", "b12"]):
            return "you may feel numbness, tiredness, or memory issues"
        elif any(w in combined for w in ["potassium"]):
            return "you may feel muscle weakness or irregular heartbeat"
        elif any(w in combined for w in ["sodium"]):
            return "you may feel dizzy, confused, or nauseous"
        elif any(w in combined for w in ["protein", "albumin"]):
            return "may indicate nutrition or liver issues"
        elif any(w in combined for w in ["thyroid", "tsh", "t3", "t4"]):
            return "your metabolism and energy levels may be affected"
        else:
            return "this is below the healthy range — ask your doctor about it"
    else:  # High
        if any(w in combined for w in ["white blood", "wbc", "neutrophil"]):
            return "your body may be fighting an infection"
        elif any(w in combined for w in ["eosinophil", "allerg", "parasit"]):
            return "you may have allergies or a parasitic infection"
        elif any(w in combined for w in ["cholesterol", "ldl", "triglyceride", "fat"]):
            return "higher risk of heart problems over time"
        elif any(w in combined for w in ["sugar", "glucose", "hba1c", "diabetes"]):
            return "may indicate diabetes or pre-diabetes"
        elif any(w in combined for w in ["creatinine", "kidney", "urea", "bun", "egfr"]):
            return "your kidneys may need attention"
        elif any(w in combined for w in ["liver", "sgpt", "sgot", "bilirubin", "alt", "ast"]):
            return "your liver may be under stress"
        elif any(w in combined for w in ["thyroid", "tsh"]):
            return "your thyroid may not be working properly"
        elif any(w in combined for w in ["uric acid", "gout"]):
            return "risk of gout (painful joints)"
        elif any(w in combined for w in ["potassium"]):
            return "could affect your heart rhythm"
        elif any(w in combined for w in ["sodium"]):
            return "may cause swelling or high blood pressure"
        elif any(w in combined for w in ["platelet", "clot"]):
            return "your blood may clot too easily"
        elif any(w in combined for w in ["esr", "crp", "inflam"]):
            return "there may be inflammation in your body"
        elif any(w in combined for w in ["red blood", "rbc", "hematocrit"]):
            return "your blood may be too thick — stay hydrated"
        elif any(w in combined for w in ["basophil"]):
            return "may indicate an allergic reaction or inflammation"
        else:
            return "this is above the healthy range — ask your doctor about it"


def generate_ai_summary(results, param_map):
    if not results:
        return "No valid parameters found."

    total = len(results)
    normal_count = sum(1 for r in results if r.get("status") == "Normal")
    concern_count = total - normal_count

    if concern_count == 0:
        return (f"Great news! All {total} test results are within the healthy range. "
                f"Your body seems to be working well. Keep up your healthy habits!")

    # Build patient-friendly explanation for each abnormal result
    lines = []
    lines.append(f"Your report checked **{total}** things in your blood. "
                 f"**{normal_count}** are perfectly fine. "
                 f"However, **{concern_count}** need your doctor's attention:")
    lines.append("")  # blank line before list

    for r in results:
        param = r.get("parameter", "")
        value = r.get("value", "")
        unit = r.get("unit", "")
        status = r.get("status", "Normal")
        if status == "Normal":
            continue

        # Get simple explanation
        simple = get_simple_explanation(param)
        
        # Get health impact (smart category-based)
        meta = param_meta.get(param, {})
        raw_meaning = meta.get("meaning", "")
        impact = _get_health_impact(param, status, raw_meaning)

        param_display = param.upper()
        status_icon = "🔴" if status == "High" else "🟡"
        status_word = "High" if status == "High" else "Low"
        
        if simple:
            lines.append(f"{status_icon} **{param_display}** ({simple})")
        else:
            lines.append(f"{status_icon} **{param_display}**")
        lines.append(f"- Value: **{value} {unit}** — **{status_word}**")
        lines.append(f"- What this means: {impact}")
        lines.append("")  # blank line between items

    lines.append("---")
    lines.append(f"✅ The other **{normal_count}** results are all healthy.")
    lines.append("")
    lines.append("👨‍⚕️ **Please share this report with your doctor for proper guidance.**")

    return "\n".join(lines)



@st.cache_data
def build_param_metadata():
    meta = {}
    for p, info in reference_ranges.items():
        full_form = ""
        meaning = ""
        candidates_to_try = [p] + (info.get("synonyms", []) or [])

        # 1) Glossary is the primary merged source (contains all lab + general medical definitions)
        if not meaning:
            for candidate in candidates_to_try:
                candidate_lower = candidate.lower()
                if candidate_lower in glossary_map:
                    raw = glossary_map[candidate_lower]
                    if raw and raw.strip():
                        meaning = raw.strip()
                        if not full_form:
                            full_form = candidate
                        break

        # 2) Try medical_corpus.json for full_form names + Hindi/Marathi translations
        marathi = ""
        hindi = ""
        if not meaning or not full_form:
            for cand in candidates_to_try:
                corpus_entry = medical_corpus.get(cand.lower())
                if corpus_entry:
                    eng = corpus_entry.get("english", "")
                    if eng and not full_form:
                        full_form = eng
                    marathi = corpus_entry.get("marathi", "")
                    hindi = corpus_entry.get("hindi", "")
                    break
        # Also fetch translations even if full_form already found
        if not marathi or not hindi:
            for cand in candidates_to_try:
                corpus_entry = medical_corpus.get(cand.lower())
                if corpus_entry:
                    if not marathi:
                        marathi = corpus_entry.get("marathi", "")
                    if not hindi:
                        hindi = corpus_entry.get("hindi", "")
                    break

        # 3) Try medical_jargon.json (broadest coverage — medications, abbreviations, conditions)
        if not meaning:
            for candidate in candidates_to_try:
                candidate_lower = candidate.lower()
                raw = medical_jargon.get(candidate_lower, "")
                if raw:
                    candidate_meaning = _sanitize_meaning(raw)
                    if candidate_meaning:
                        meaning = candidate_meaning
                        if not full_form:
                            full_form = candidate
                        break

        if not full_form:
            full_form = p.upper()
        if not meaning:
            meaning = ""

        meta[p] = {"full_form": full_form, "meaning": meaning, "marathi": marathi, "hindi": hindi}
    return meta


param_meta = build_param_metadata()

# OCR helpers

def preprocess_image(img):
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)
# ---------- replace extract_text ----------
def extract_text(file_path):
    """
    Preserve line breaks from OCR; return text with newlines preserved.
    Also return a list of lines (cleaned) for downstream local matching.
    """
    text = ""
    try:
        if file_path.lower().endswith(".pdf"):
            pages = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)
            page_texts = []
            for page in pages:
                processed = preprocess_image(page)
                page_t = pytesseract.image_to_string(processed, config="--psm 6")
                page_texts.append(page_t.rstrip())
            text = "\n\n".join(page_texts)
        elif file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif file_path.lower().endswith((".html", ".htm")):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
            # Strip style/script blocks
            text = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL|re.IGNORECASE)
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL|re.IGNORECASE)
            # Insert newlines at block-level element boundaries (tr, div, p, br, li, h1-h6, table)
            text = re.sub(r'<(?:tr|div|p|br|li|h[1-6]|table|/tr|/table)[^>]*>', '\n', text, flags=re.IGNORECASE)
            # Replace td/th with tab to preserve column separation on same row
            text = re.sub(r'<(?:td|th)[^>]*>', '\t', text, flags=re.IGNORECASE)
            # Remove remaining HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Decode HTML entities
            text = re.sub(r'&nbsp;|&#xa0;|&#160;', ' ', text, flags=re.IGNORECASE)
            text = re.sub(r'&[a-zA-Z]+;', ' ', text)
            text = re.sub(r'&#\d+;', ' ', text)
            # Collapse spaces (but keep tabs and newlines)
            text = re.sub(r'[ ]+', ' ', text)
            text = re.sub(r'\t', '  ', text)  # convert tabs to double-space
        else:
            img = Image.open(file_path)
            processed = preprocess_image(img)
            text = pytesseract.image_to_string(processed, config="--psm 6")

    except Exception as e:
        st.error(f"OCR failed: {e}")

    # Normalize line endings and keep them
    # remove excessive blank lines but keep one newline per break
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    clean_text = "\n".join(lines)
    return clean_text


# Numeric normalization

# ---------- improved normalize_value ----------
def normalize_value(value_str, param=None):
    """
    More cautious numeric extraction:
    - Handles < and >
    - Handles ranges by averaging
    - Rejects values that look like dates (e.g., dd-mm-yyyy or yyyy-mm-dd)
    - Returns None for suspicious numbers
    """
    try:
        if value_str is None:
            return None
        s = str(value_str).strip()
        if not s:
            return None
        # Handle comma: thousands separator (e.g., "9,000", "2,28,000") vs decimal (e.g., "1,5")
        # If ALL commas are followed by 2-3 digits (Indian or Western format), treat as thousands separators
        if re.match(r'^\d{1,3}(,\d{2,3})+$', s):
            # Pure comma-separated number (Indian: 2,28,000 or Western: 228,000)
            s = s.replace(",", "")
        else:
            s = re.sub(r',(\d{3})(?!\d)', r'\1', s)  # "9,000" -> "9000", "16,700" -> "16700"
            s = s.replace(",", ".")  # remaining commas are decimal: "1,5" -> "1.5"

        # quick date-like rejection: tokens with two dashes or slashes and 3-4 digit year
        if re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", s) or re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", s):
            return None

        # range  e.g. "1.2-1.5" or "1.2 - 1.5"
        if "-" in s and re.search(r"\d", s):
            parts = re.split(r"\s*-\s*", s)
            nums = []
            for p in parts:
                m = re.search(r"[<>]?\s*(\d+(?:\.\d+)?)", p)
                if m:
                    nums.append(float(m.group(1)))
            if nums:
                val = sum(nums) / len(nums)
            else:
                return None

        elif s.startswith("<") or s.startswith("≤"):
            m = re.search(r"[<≤]\s*(\d+(?:\.\d+)?)", s)
            if not m:
                return None
            # slightly below the reported threshold
            val = float(m.group(1)) * 0.98

        elif s.startswith(">") or s.startswith("≥"):
            m = re.search(r"[>≥]\s*(\d+(?:\.\d+)?)", s)
            if not m:
                return None
            val = float(m.group(1)) * 1.02

        else:
            # find the first standalone numeric token (not part of a longer ID)
            m = re.search(r"(?<!\d)(\d{1,7}(?:\.\d+)?)(?!\d)", s)
            if not m:
                return None
            val = float(m.group(1))

        # ignore impossible or suspicious lab values
        if val <= 0 or val > 1e6:
            return None

        # optional sanity check against reference ranges if available
        if param and param in reference_ranges:
            low = reference_ranges[param].get("low")
            high = reference_ranges[param].get("high")
            unit = reference_ranges[param].get("unit", "")
            if low is not None and high is not None:
                # For percentage-based params (max 100%), reject >100
                if "%" in unit and val > 100:
                    return None
                # For non-percentage params, reject values absurdly larger than range
                # Use 100x to allow legitimately high pathological values (e.g., CRP 267 with range 0-5)
                if "%" not in unit and val > 100 * max(1.0, high):
                    return None
                # OCR decimal-drop correction: if value far exceeds range but value/10, /100, or /1000 fits,
                # assume OCR dropped the decimal point (e.g., "3.9" read as "39", "3.42" read as "342")
                for divisor in [10, 100, 1000]:
                    if val > high * 3 and val / divisor >= low * 0.5 and val / divisor <= high * 2:
                        val = val / divisor
                        break

        return val
    except Exception:
        return None


# Parameter extraction logic (line-level + local window)

import json

def detect_report_type(text):
    """Detect report type(s) by matching section names and their parameter synonyms from reference_ranges.csv"""
    text_lower = text.lower()
    # Normalize OCR artifacts: remove dashes, parentheses, extra spaces for better matching
    text_normalized = re.sub(r'[()/:;,]', ' ', text_lower)
    text_normalized = re.sub(r'\s*-\s*', ' ', text_normalized)
    text_normalized = re.sub(r'\s+', ' ', text_normalized)
    scores = {}
    
    for section_name, params in section_map.items():
        score = 0
        # Check if section name keywords appear in the text
        section_words = section_name.split()
        for word in section_words:
            if len(word) > 2 and word in text_normalized:
                score += 2
        
        # Check how many parameters/synonyms from this section appear in text
        for param in params:
            param_normalized = re.sub(r'\s*-\s*', ' ', param)
            if param_normalized in text_normalized or param in text_lower:
                score += 1
            details = reference_ranges.get(param, {})
            for syn in details.get("synonyms", []):
                syn_normalized = re.sub(r'\s*-\s*', ' ', syn)
                if syn_normalized in text_normalized or syn in text_lower:
                    score += 1
                    break
        
        if score >= 2:
            scores[section_name] = score
    
    if not scores:
        return []
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

def get_allowed_params(detected_types):
    """Get set of allowed parameters based on detected report sections."""
    if not detected_types:
        return None  # None means allow everything
    
    allowed = set()
    for section in detected_types:
        allowed.update(section_map.get(section, []))
    return allowed

def extract_parameters(text, reference_ranges, fuzzy=False, fuzz_threshold=90, char_window=40):
    """
    Search each line separately; for each parameter occurrence, find numeric value
    within a small character window to reduce false positives.
    """
    results = []
    seen_params = set()
    
    # Debug log
    _debug_log = []
    
    # Detect report type(s) for urine-specific logic only
    detected_types = detect_report_type(text)
    is_urine = any("urine" in dt for dt in detected_types)
    
    # split into lines for local matching
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    _debug_log.append(f"Lines: {len(lines)}, detected_types: {detected_types}")

    # Pre-process: normalize lines by inserting spaces before uppercase letters in
    # concatenated words (e.g., WBCCOUNT -> WBC COUNT, RBCCOUNT -> RBC COUNT)
    lines_normalized = []
    for line in lines:
        # Insert space between lowercase/uppercase or between known patterns
        normalized = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
        # Also handle all-caps concatenation like WBCCOUNT, RBCCOUNT, PLATELETCOUNT
        normalized = re.sub(r'([A-Z]+)(COUNT|CRIT|PHILS|CYTES)', r'\1 \2', normalized)
        lines_normalized.append(normalized)

    # Pre-compute aggressively normalized lines (strip dashes, parentheses, slashes, colons, periods, extra spaces)
    lines_cleaned = []
    for line in lines:
        cleaned = line.lower()
        # Remove periods between single letters (R.B.C. -> RBC, H.C.T. -> HCT, E.S.R. -> ESR)
        cleaned = re.sub(r'(?<![a-z0-9])([a-z])\.([a-z])\.([a-z])\.?(?![a-z0-9])', r'\1\2\3', cleaned)
        cleaned = re.sub(r'(?<![a-z0-9])([a-z])\.([a-z])\.?(?![a-z0-9])', r'\1\2', cleaned)
        # Remove parenthetical abbreviations — keep both the full name and abbreviation
        # e.g. "packed cell volume (hct)" → "packed cell volume hct"
        cleaned = re.sub(r'[()/:;,\-]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        lines_cleaned.append(cleaned)

    for li, line in enumerate(lines):
        line_lower = line.lower()
        # Also check the normalized version
        norm_lower = lines_normalized[li].lower()
        cleaned_lower = lines_cleaned[li]
        
        # Log lines that contain key CBC terms
        if any(kw in line_lower for kw in ['rbc','mcv','mch','wbc','neutro','lymph','platelet','mpv','hemoglobin','hematocrit','rdw','eosino','mono','baso']):
            _debug_log.append(f"[{li}] LINE: {line[:80]}")
        
        for param, details in reference_ranges.items():
            if param in seen_params:
                continue

            candidates = [param] + details.get("synonyms", [])
            matched = False
            val = None

            for cand in candidates:
                cand_lower = cand.lower()
                # Skip very short candidates (1-2 chars) — too many false positives
                if len(cand_lower) < 3:
                    continue
                # Aggressively cleaned candidate (strip dashes, parentheses, slashes, etc.)
                cand_cleaned = re.sub(r'[()/:;,\-]', ' ', cand_lower)
                cand_cleaned = re.sub(r'\s+', ' ', cand_cleaned).strip()
                cand_nospace = cand_lower.replace(" ", "").replace("-", "")

                # Try matching in order: original, normalized, cleaned (aggressive), nospace
                match_in_original = re.search(rf"\b{re.escape(cand_lower)}\b", line_lower)
                match_in_normalized = re.search(rf"\b{re.escape(cand_lower)}\b", norm_lower)
                match_cleaned = re.search(rf"\b{re.escape(cand_cleaned)}\b", cleaned_lower) if cand_cleaned != cand_lower else None
                match_nospace = re.search(rf"\b{re.escape(cand_nospace)}\b", line_lower.replace(" ", "")) if " " in cand_lower or "-" in cand_lower else None

                if match_in_original or match_in_normalized or match_cleaned or match_nospace:
                    matched = True
                    # Determine which line variant matched and get position
                    if match_in_original:
                        m_word = match_in_original
                        search_line = line
                    elif match_in_normalized:
                        m_word = match_in_normalized
                        search_line = lines_normalized[li]
                    elif match_cleaned:
                        m_word = match_cleaned
                        search_line = lines_cleaned[li]
                    else:
                        m_word = match_nospace
                        # Use the nospace version of the line so positions align
                        search_line = line_lower.replace(" ", "")

                    # Search for value AFTER the parameter name (not before) to avoid picking up wrong numbers
                    end_of_match = m_word.end()
                    right = min(len(search_line), end_of_match + char_window + 20)
                    window = search_line[end_of_match:right]

                    # search for numeric pattern inside window (supports Indian number format like 2,28,000)
                    m_val = re.search(r"([<>≤≥]?\s*\d[\d,]*\.?\d*(?:\s*-\s*\d[\d,]*\.?\d*)?)", window)
                    if m_val:
                        candidate_val = m_val.group(1)
                        val = normalize_value(candidate_val, param)
                    
                    # Multi-line fallback: if no value on same line, check next 2 lines
                    # (OCR of tabular PDFs sometimes puts param names and values on separate lines)
                    if val is None:
                        for offset in range(1, 3):
                            if li + offset < len(lines):
                                next_line = lines[li + offset]
                                m_next = re.search(r"^\s*([<>≤≥]?\s*\d[\d,]*\.?\d*)\s*", next_line)
                                if m_next:
                                    val = normalize_value(m_next.group(1), param)
                                    if val is not None:
                                        break
                    break

                # fuzzy match option: compare candidate to the line (partial)
                if fuzzy:
                    score = rfuzz.partial_ratio(cand_lower, line_lower)
                    if score >= fuzz_threshold:
                        matched = True
                        # same logic: find approximate position using simple find
                        idx = line_lower.find(cand_lower)
                        if idx == -1:
                            idx = 0
                        left = max(0, idx - char_window)
                        right = min(len(line), idx + len(cand_lower) + char_window)
                        window = line[left:right]
                        m_val = re.search(r"([<>≤≥]?\s*\d[\d,]*\.?\d*(?:\s*-\s*\d[\d,]*\.?\d*)?)", window)
                        if m_val:
                            candidate_val = m_val.group(1)
                            val = normalize_value(candidate_val, param)
                        break

            if matched and val is not None:
                low, high = details.get("low"), details.get("high")
                status = "Normal"
                if low is not None and val < low:
                    status = "Low"
                elif high is not None and val > high:
                    status = "High"
                _debug_log.append(f"  MATCH: {param}={val} ({status}) via '{cand_lower}' on line {li}")

                results.append({
                    "parameter": param,
                    "value": val,
                    "unit": details.get("unit", ""),
                    "status": status,
                    "line_index": li
                })
                seen_params.add(param)
            
            elif matched and val is None:
                _debug_log.append(f"  NO-VAL: {param} matched '{cand_lower}' on line {li} but val=None")
            
            # Handle qualitative results (Present/Absent/Positive/Negative/Nil/Trace)
            if matched and val is None and is_urine:
                qual_match = re.search(
                    r"(present\s*\([+]+\)|present|absent|positive|negative|nil|trace|s[\.\s]*turbid)",
                    line_lower
                )
                if qual_match:
                    qual_val = qual_match.group(1).strip()
                    if any(k in qual_val for k in ("present", "positive", "turbid")):
                        status = "Abnormal"
                        display_val = qual_val.upper()
                    elif "trace" in qual_val:
                        status = "Normal"
                        display_val = "Trace"
                    else:
                        status = "Normal"
                        display_val = "Absent"
                    results.append({
                        "parameter": param,
                        "value": display_val,
                        "unit": details.get("unit", ""),
                        "status": status,
                        "line_index": li
                    })
                    seen_params.add(param)

    # Write debug log
    _debug_log.append(f"Total results: {len(results)}")
    for r in results:
        _debug_log.append(f"  {r['parameter']} = {r['value']} ({r['status']})")
    try:
        with open("_extraction_debug.log", "w", encoding="utf-8") as f:
            f.write("\n".join(_debug_log))
    except:
        pass
    
    return results

# Build patient-friendly markdown (English) from results

STATUS_EMOJI = {"Normal": "🟢", "High": "🔴", "Low": "🟡", "Abnormal": "🔴"}

def build_patient_markdown(results, lang="en"):
    if not results:
        if lang == "mr":
            return "तुमच्या अहवालात कोणतेही वैध चाचणी निकाल आढळले नाहीत."
        elif lang == "hi":
            return "आपकी रिपोर्ट में कोई मान्य परीक्षण परिणाम नहीं मिला."
        return "No valid test results detected in your report."
    blocks = []
    for r in results:
        p = r["parameter"]
        meta = param_meta.get(p, {"full_form": p.upper(), "meaning": "", "marathi": "", "hindi": ""})
        full = meta["full_form"]
        # Use BART model to simplify meaning into plain English
        meaning = get_simple_explanation(p) or meta["meaning"] or "Meaning not available."
        marathi_name = meta.get("marathi", "")
        hindi_name = meta.get("hindi", "")
        emoji = STATUS_EMOJI.get(r["status"], "ℹ️")
        value = r["value"]
        unit = r.get("unit", "")
        status = r["status"]

        if lang == "mr":
            # Marathi output
            local_name = marathi_name or full
            meaning_display = marathi_name if marathi_name else meaning
            if status == "High" or status == "Abnormal":
                action = "हे संभाव्य आरोग्य समस्या दर्शवू शकते. कृपया डॉक्टरांचा सल्ला घ्या."
                status_mr = "जास्त" if status == "High" else "असामान्य"
            elif status == "Low":
                action = "हे सामान्य श्रेणीपेक्षा कमी असू शकते. सल्ला घेणे योग्य आहे."
                status_mr = "कमी"
            else:
                action = "हे सामान्य श्रेणीत आहे, चांगल्या आरोग्याचे लक्षण."
                status_mr = "सामान्य"
            header = f"{emoji} **{p.upper()} ({local_name})**"
            block_lines = [
                header,
                f"- **अर्थ:** {meaning_display}",
                f"- **निकाल:** {value} {unit} — **{status_mr}**",
                f"- **सल्ला:** {action}"
            ]
        elif lang == "hi":
            # Hindi output
            local_name = hindi_name or full
            meaning_display = hindi_name if hindi_name else meaning
            if status == "High" or status == "Abnormal":
                action = "यह संभावित स्वास्थ्य समस्या का संकेत हो सकता है. कृपया डॉक्टर से परामर्श करें."
                status_hi = "अधिक" if status == "High" else "असामान्य"
            elif status == "Low":
                action = "यह सामान्य सीमा से कम हो सकता है. परामर्श लेना उचित है."
                status_hi = "कम"
            else:
                action = "यह सामान्य सीमा में है, अच्छे स्वास्थ्य का संकेत."
                status_hi = "सामान्य"
            header = f"{emoji} **{p.upper()} ({local_name})**"
            block_lines = [
                header,
                f"- **अर्थ:** {meaning_display}",
                f"- **परिणाम:** {value} {unit} — **{status_hi}**",
                f"- **सलाह:** {action}"
            ]
        else:
            # English output
            if status == "High" or status == "Abnormal":
                action = "This may indicate a possible health issue. Please consult your doctor."
            elif status == "Low":
                action = "This may be below the normal range. Consultation is advised."
            else:
                action = "This is within the normal range, indicating good health."
            header = f"{emoji} **{p.upper()} ({full})**"
            block_lines = [
                header,
                f"- **Meaning:** {meaning}",
                f"- **Result:** {value} {unit} — **{status}**",
                f"- **Advice:** {action}"
            ]
        blocks.append("\n".join(block_lines))
    md = "\n\n".join(blocks)
    return md


def translate_with_fallback(text, target_lang):
    url = "https://google-translate113.p.rapidapi.com/api/v1/translator/html"
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "82aa2b1fbbmsh4ef73325297a2f5p1bea3bjsnba30f3e7d41a",
        "x-aibit-key": "5cf048c0-13ba-11ee-a37b-d799f0284f13"
    }
    payload = {"from": "auto", "to": target_lang, "html": text}
    try:
        r = requests.post(url, json=payload, headers=headers)
        if r.status_code == 200:
            data = r.json()
            return data.get("trans", text)
        else:
            st.warning(f"Translation API error: {r.status_code}")
            return text
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text


# Sidebar controls

with st.sidebar:
    st.header("Options")
    language = st.radio(
        "Choose summary language (only summary will be shown):",
        ("English", "Hindi", "Marathi")
    )
    debug_opt = st.checkbox("Show OCR & debug info", value=False)
    st.markdown("---")
    st.write("Tips:")
    st.write("- Upload a clear PDF or image file.")
    st.write("- Check that the report is legible and not folded or blurred for best OCR results.")

# Short summary function

def build_short_summary(results):
    if not results:
        return "No test results detected."

    total = len(results)
    high_count = sum(1 for r in results if r.get("status") == "High")
    low_count = sum(1 for r in results if r.get("status") == "Low")
    abnormal_count = sum(1 for r in results if r.get("status") == "Abnormal")
    normal_count = total - high_count - low_count - abnormal_count
    concern_count = high_count + low_count + abnormal_count

    if concern_count == 0:
        return f"✅ All {total} test results are normal. No health concerns found."

    parts = []
    if low_count > 0:
        parts.append(f"{low_count} low")
    if high_count > 0:
        parts.append(f"{high_count} high")
    if abnormal_count > 0:
        parts.append(f"{abnormal_count} abnormal")

    concern_text = ", ".join(parts)
    return f"⚠️ {concern_count} out of {total} results need attention ({concern_text}). {normal_count} results are normal. Please consult your doctor for the abnormal values."

# File uploader

uploaded_file = st.file_uploader("📂 Upload your medical report", type=["pdf","png","jpg","jpeg","txt","html","htm"])

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(uploaded_file.read())
        temp_path = tf.name

    try:
        with st.spinner("🧠 Running OCR and extracting parameters..."):
            raw_text = extract_text(temp_path)
        if debug_opt:
            st.subheader("🔍 Raw OCR text")
            st.text_area("OCR Text", raw_text, height=300)
            st.write(f"Text length: {len(raw_text)} chars, Lines: {len(raw_text.splitlines())}")

        # Extract parameters
        params = extract_parameters(raw_text, reference_ranges, fuzzy=False)
        if debug_opt:
            st.write(f"Extraction result: {len(params)} params found (fuzzy=False)")
        if not params:
            params = extract_parameters(raw_text, reference_ranges, fuzzy=True, fuzz_threshold=85)
        param_to_full = {p: v["full_form"] for p, v in param_meta.items()}
        ai_summary = generate_ai_summary(params, param_to_full)



        if not params:
            st.warning("No parameters detected.")
        else:
            st.subheader("📊 Extracted Parameters")
            df = pd.DataFrame(params)
            df_display = df[["parameter","value","unit","status"]].rename(columns={
                "parameter": "Parameter",
                "value": "Value",
                "unit": "Unit",
                "status": "Status",
            })
            st.dataframe(df_display, use_container_width=True)

        # Short summary at top (quick overview)
        short_summary = build_short_summary(params)
        st.subheader("📝 Quick Summary")
        st.markdown(f"**{short_summary}**")

        # Detailed patient-friendly breakdown
        lang_code = "en" if language=="English" else ("hi" if language=="Hindi" else "mr")
        patient_md = build_patient_markdown(params, lang=lang_code)
        translated_summary = translate_with_fallback(patient_md, lang_code)

        st.subheader("💬 Patient-Friendly Summary")
        st.markdown(translated_summary, unsafe_allow_html=True)

        # AI Summary at bottom (most important, detailed explanation)
        st.subheader("💡 AI-Generated Summary")
        st.markdown(ai_summary, unsafe_allow_html=True)

        # Always show Hindi and Marathi translations below English
        st.markdown("---")
        st.subheader("🇮🇳 Hindi Summary (हिंदी)")
        hindi_ai = translate_with_fallback(ai_summary, "hi")
        st.markdown(hindi_ai, unsafe_allow_html=True)

        st.subheader("🇮🇳 Marathi Summary (मराठी)")
        marathi_ai = translate_with_fallback(ai_summary, "mr")
        st.markdown(marathi_ai, unsafe_allow_html=True)

        st.download_button(
            "📥 Download Patient Summary",
            data=translated_summary,
            file_name=f"patient_summary_{language.lower()}.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Processing failed: {e}")
        st.error(traceback.format_exc())
    finally:
        try:
            os.remove(temp_path)
        except:
            pass

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Empowering Patients with Clear, Multilingual, and Personalized Health Insights.")

###to run the command

####streamlit run main.py
