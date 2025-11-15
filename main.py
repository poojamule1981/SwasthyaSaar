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

POPPLER_PATH = r"C:\Program Files\poppler-25.07.0\Library\bin"  
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  
FINE_TUNED_MODEL_PATH = "./models/lab_summarizer"  

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
    except Exception as e:
        st.warning(f"Failed to load glossary: {e}")
    return glossary

@st.cache_data
def load_readme_exp():
    fname = find_file("readme_exp.csv")
    fallback = {}
    if not fname:
        return fallback
    try:
        df = pd.read_csv(fname, engine="python")
        ann_col, gpt_col = None, None
        for c in df.columns:
            cl = c.lower().strip()
            if cl == "ann_text":
                ann_col = c
            if cl in ("gpt_generated", "gpt_text_to_annotate"):
                gpt_col = c
        if ann_col is None:
            ann_col = df.columns[0]
        if gpt_col is None and df.shape[1] > 1:
            gpt_col = df.columns[1]
        for _, row in df.iterrows():
            key = str(row[ann_col]).strip().lower()
            val = str(row[gpt_col]).strip() if gpt_col else ""
            if key:
                fallback[key] = val
    except Exception as e:
        st.warning(f"Failed to load readme_exp.csv: {e}")
    return fallback

@st.cache_data
def load_reference_ranges():
    fname = find_file("reference_ranges.csv")
    reference = {}
    if not fname:
        return reference
    try:
        df = pd.read_csv(fname, engine="python")
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
                reference[p] = {"low": low, "high": high, "unit": unit, "synonyms": syns}
    except Exception as e:
        st.error(f"Failed to load reference_ranges.csv: {e}")
    return reference

glossary_map = load_glossary()
readme_map = load_readme_exp()
reference_ranges = load_reference_ranges()


def _sanitize_meaning(text):
    """
    Clean a meaning string:
     - unescape HTML entities
     - remove newlines / excessive whitespace
     - reject extremely long or clearly model-like strings
    """
    if not text:
        return ""
    s = html.unescape(str(text)).strip()
    s = re.sub(r"\s+", " ", s)
    # If the string looks like a long paragraph (likely noisy) or contains 'gpt' / 'ai' tokens, reject
    if len(s) > 160 or re.search(r"\bgpt\b|\bchatgpt\b|\bmodel\b|\btranslate\b", s, re.I):
        return ""
    # remove suspicious fragments like 'ISS and accuchecks' that are not definitional — heuristic:
    if re.search(r"(plan to|accuchecks|contact us|please consult|A1c dietary)", s, re.I):
        return ""
    # Ensure it is short — keep first sentence
    if "." in s:
        s = s.split(".")[0].strip() + "."
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
def generate_ai_summary(results, param_map):
    if not results:
        return "No valid parameters found."

    # build base text
    base = " ".join([f"{r['parameter'].capitalize()}: {r['value']} {r['unit']} ({r['status']})." for r in results])
    expanded = expand_abbreviations_for_summary(base, param_map)

    # summarize using cached pipeline
    try:
        out = summarizer(expanded, max_length=120, min_length=30, do_sample=False)
        return out[0]["summary_text"]
    except Exception:
        # fallback: return expanded text if summarization fails
        return expanded



@st.cache_data
def build_param_metadata():
    meta = {}
    for p, info in reference_ranges.items():
        full_form = ""
        meaning = ""
        syns = info.get("synonyms", [])
        candidate = None
        for s in syns:
            if " " in s:
                candidate = s
                break
        if not candidate and syns:
            candidate = syns[0]

        # 1) Prefer glossary_map (trusted CSV)
        if candidate and candidate.lower() in glossary_map:
            meaning = _sanitize_meaning(glossary_map[candidate.lower()])
            full_form = candidate
        if not meaning and p.lower() in glossary_map:
            meaning = _sanitize_meaning(glossary_map[p.lower()])
            if not full_form:
                full_form = p

        # 2) Only use readme_map as a last resort and only if short & clean
        if not meaning:
            key = (candidate or p).lower()
            raw = readme_map.get(key, "")
            candidate_meaning = _sanitize_meaning(raw)
            if candidate_meaning:
                meaning = candidate_meaning
                if not full_form:
                    full_form = candidate or p

        # 3) final safe fallbacks
        if not full_form:
            full_form = p.upper()
        if not meaning:
            # Use a short generic template instead of long junk
            meaning = f"{full_form} is a routine medical parameter; consult your clinician for details."

        meta[p] = {"full_form": full_form, "meaning": meaning}
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
        s = s.replace(",", ".")  # allow comma as decimal

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
            if low is not None and high is not None:
                # if value is absurdly larger than expected, flag as None
                if val > 10 * max(1.0, high):
                    return None

        return val
    except Exception:
        return None


# Parameter extraction logic (line-level + local window)

def extract_parameters(text, reference_ranges, fuzzy=False, fuzz_threshold=90, char_window=40):
    """
    Search each line separately; for each parameter occurrence, find numeric value
    within a small character window to reduce false positives.
    """
    results = []
    seen_params = set()
    # split into lines for local matching
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for li, line in enumerate(lines):
        line_lower = line.lower()
        for param, details in reference_ranges.items():
            if param in seen_params:
                continue

            candidates = [param] + details.get("synonyms", [])
            matched = False
            val = None

            for cand in candidates:
                cand_lower = cand.lower()
                # exact word match in this line
                if re.search(rf"\b{re.escape(cand_lower)}\b", line_lower):
                    matched = True
                    # find the first occurrence position
                    m_word = re.search(rf"\b{re.escape(cand_lower)}\b", line_lower)
                    start = m_word.start()
                    # create a substring window around the match (same line)
                    left = max(0, start - char_window)
                    right = min(len(line), start + len(cand_lower) + char_window)
                    window = line[left:right]

                    # search for numeric pattern inside window
                    m_val = re.search(r"([<>≤≥]?\s*\d+[.,]?\d*(?:\s*-\s*\d+[.,]?\d*)?)", window)
                    if m_val:
                        candidate_val = m_val.group(1)
                        val = normalize_value(candidate_val, param)
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
                        m_val = re.search(r"([<>≤≥]?\s*\d+[.,]?\d*(?:\s*-\s*\d+[.,]?\d*)?)", window)
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

                results.append({
                    "parameter": param,
                    "value": val,
                    "unit": details.get("unit", ""),
                    "status": status,
                    "line_index": li
                })
                seen_params.add(param)

    return results

# Build patient-friendly markdown (English) from results

STATUS_EMOJI = {"Normal": "🟢", "High": "🔴", "Low": "🟡"}

def build_patient_markdown(results):
    if not results:
        return "No valid test results detected in your report."
    blocks = []
    for r in results:
        p = r["parameter"]
        meta = param_meta.get(p, {"full_form": p.upper(), "meaning": f"{p.upper()} is a routine medical test."})
        full = meta["full_form"]
        meaning = meta["meaning"]
        emoji = STATUS_EMOJI.get(r["status"], "ℹ️")
        value = r["value"]
        unit = r.get("unit", "")
        status = r["status"]
        if status == "High":
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

    categories = {"Normal": [], "High": [], "Low": []}
    for r in results:
        status = r.get("status", "Normal")
        param = r.get("parameter", "").upper()
        categories[status].append(param)

    # Build summary string
    parts = []
    for cat in ["Normal", "High", "Low"]:
        if categories[cat]:
            names = ", ".join(categories[cat])
            parts.append(f"{cat}: {names} ({len(categories[cat])})")

    summary_text = " | ".join(parts)
    return summary_text

# File uploader

uploaded_file = st.file_uploader("📂 Upload your medical report", type=["pdf","png","jpg","jpeg","txt"])

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

        # Extract parameters
        params = extract_parameters(raw_text, reference_ranges, fuzzy=False)
        if not params:
            params = extract_parameters(raw_text, reference_ranges, fuzzy=True, fuzz_threshold=85)
        param_to_full = {p: v["full_form"] for p, v in param_meta.items()}
        ai_summary = generate_ai_summary(params, param_to_full)



        st.subheader("📊 Extracted Parameters")
        if params:
            df = pd.DataFrame(params)
            df_display = df[["parameter","value","unit","status"]].rename(columns={
                "parameter": "Parameter",
                "value": "Value",
                "unit": "Unit",
                "status": "Status",
               
            })
            st.dataframe(df_display, use_container_width=True)
        else:
            st.warning("No parameters detected.")

            

        st.subheader("💡 AI-Generated Summary")
        st.markdown(ai_summary, unsafe_allow_html=True)

        # Detailed patient summary
        patient_md = build_patient_markdown(params)
        lang_code = "en" if language=="English" else ("hi" if language=="Hindi" else "mr")
        translated_summary = translate_with_fallback(patient_md, lang_code)

        st.subheader("💬 Patient-Friendly Summary")
        st.markdown(translated_summary, unsafe_allow_html=True)

        # Short summary
        short_summary = build_short_summary(params)
        translated_short_summary = translate_with_fallback(short_summary, lang_code)
        st.markdown("📝 Short Summary")
        st.markdown(translated_short_summary, unsafe_allow_html=True)




        st.subheader("📝 Short Summary")
        st.markdown(f"**{translated_short_summary }**")

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
