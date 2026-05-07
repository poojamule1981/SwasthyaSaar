"""
Generate high-quality training data from lab report images.

Steps:
1. OCR each image using Tesseract
2. Extract medical parameters using our reference_ranges logic
3. Generate a clean, structured medical summary from extracted params
4. Save as CSV (report, summary) for BART fine-tuning
"""

import os
import sys
import re
import csv
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import cv2

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
LAB_REPORT_DIR = os.path.join(DATA_DIR, "lab_report")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "lab_reports_dataset.csv")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ===== Load reference ranges =====
def load_reference_ranges():
    fname = os.path.join(DATA_DIR, "reference_ranges.csv")
    reference = {}
    if not os.path.exists(fname):
        print(f"ERROR: {fname} not found")
        return reference
    current_section = "general"
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                current_section = line.lstrip("# ").strip().lower()
            elif line and not line.startswith("parameter"):
                parts = line.split(",")
                if len(parts) >= 4:
                    param = parts[0].strip().lower()
                    try:
                        low = float(parts[1]) if parts[1].strip() else None
                    except ValueError:
                        low = None
                    try:
                        high = float(parts[2]) if parts[2].strip() else None
                    except ValueError:
                        high = None
                    unit = parts[3].strip() if len(parts) > 3 else ""
                    syns = []
                    if len(parts) > 4:
                        syns = [s.strip().lower() for s in parts[4].split(";") if s.strip()]
                    reference[param] = {
                        "low": low, "high": high, "unit": unit,
                        "synonyms": syns, "section": current_section
                    }
    return reference

reference_ranges = load_reference_ranges()
print(f"Loaded {len(reference_ranges)} reference parameters")

# ===== Image preprocessing =====
def preprocess_image(img):
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

# ===== OCR =====
def ocr_image(img_path):
    try:
        img = Image.open(img_path)
        processed = preprocess_image(img)
        text = pytesseract.image_to_string(processed, config="--psm 6")
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)
    except Exception as e:
        print(f"  OCR failed for {os.path.basename(img_path)}: {e}")
        return ""

# ===== Parameter extraction (simplified version of main.py logic) =====
def normalize_value(value_str, param=None):
    try:
        if not value_str:
            return None
        s = str(value_str).strip()
        s = re.sub(r',(\d{3})(?!\d)', r'\1', s)
        s = s.replace(",", ".")
        if re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", s):
            return None
        if "-" in s and re.search(r"\d", s):
            parts = re.split(r"\s*-\s*", s)
            nums = []
            for p in parts:
                m = re.search(r"(\d+(?:\.\d+)?)", p)
                if m:
                    nums.append(float(m.group(1)))
            if nums:
                return sum(nums) / len(nums)
            return None
        m = re.search(r"(?<!\d)(\d{1,7}(?:\.\d+)?)(?!\d)", s)
        if not m:
            return None
        val = float(m.group(1))
        if val <= 0 or val > 1e6:
            return None
        # Decimal-drop correction
        if param and param in reference_ranges:
            high = reference_ranges[param].get("high")
            low = reference_ranges[param].get("low")
            unit = reference_ranges[param].get("unit", "")
            if high is not None and low is not None:
                if "%" in unit and val > 100:
                    return None
                if "%" not in unit and val > 100 * max(1.0, high):
                    return None
                if val > high * 3 and val / 10 >= low and val / 10 <= high:
                    val = val / 10
        return val
    except:
        return None

def extract_parameters(text):
    results = []
    seen = set()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Pre-compute cleaned lines
    lines_cleaned = []
    for line in lines:
        cleaned = line.lower()
        cleaned = re.sub(r'(?<![a-z0-9])([a-z])\.([a-z])\.([a-z])\.?(?![a-z0-9])', r'\1\2\3', cleaned)
        cleaned = re.sub(r'(?<![a-z0-9])([a-z])\.([a-z])\.?(?![a-z0-9])', r'\1\2', cleaned)
        cleaned = re.sub(r'[()/:;,\-]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        lines_cleaned.append(cleaned)

    for li, line in enumerate(lines):
        line_lower = line.lower()
        cleaned_lower = lines_cleaned[li]
        
        for param, details in reference_ranges.items():
            if param in seen:
                continue
            candidates = [param] + details.get("synonyms", [])
            
            for cand in candidates:
                cand_lower = cand.lower()
                if len(cand_lower) < 3:
                    continue
                cand_cleaned = re.sub(r'[()/:;,\-]', ' ', cand_lower)
                cand_cleaned = re.sub(r'\s+', ' ', cand_cleaned).strip()
                
                match = (re.search(rf"\b{re.escape(cand_lower)}\b", line_lower) or
                         re.search(rf"\b{re.escape(cand_cleaned)}\b", cleaned_lower))
                
                if match:
                    # Search for value after param name
                    end = match.end()
                    search_in = line_lower if re.search(rf"\b{re.escape(cand_lower)}\b", line_lower) else cleaned_lower
                    window = search_in[end:end+60]
                    m_val = re.search(r"(\d+[.,]?\d*)", window)
                    val = None
                    if m_val:
                        val = normalize_value(m_val.group(1), param)
                    # Multi-line fallback
                    if val is None:
                        for offset in range(1, 3):
                            if li + offset < len(lines):
                                m_next = re.search(r"^\s*(\d+[.,]?\d*)\s*", lines[li + offset])
                                if m_next:
                                    val = normalize_value(m_next.group(1), param)
                                    if val is not None:
                                        break
                    if val is not None:
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
                            "status": status
                        })
                        seen.add(param)
                    break
    return results

# ===== Generate clean summary from extracted parameters =====
def generate_clean_summary(results):
    if not results:
        return ""
    
    normal = []
    abnormal = []
    
    for r in results:
        p = r["parameter"].upper()
        v = r["value"]
        u = r["unit"]
        s = r["status"]
        
        entry = f"{p}: {v} {u}".strip()
        if s == "Normal":
            normal.append(p)
        else:
            abnormal.append(f"{entry} ({s})")
    
    parts = []
    if abnormal:
        parts.append("Abnormal findings: " + "; ".join(abnormal) + ".")
    if normal:
        parts.append("Normal parameters: " + ", ".join(normal) + ".")
    if abnormal:
        parts.append("Please consult your doctor regarding the abnormal values.")
    else:
        parts.append("All extracted parameters are within normal range.")
    
    return " ".join(parts)

# ===== Main: Process all images =====
def main():
    if not os.path.exists(LAB_REPORT_DIR):
        print(f"ERROR: Lab report folder not found: {LAB_REPORT_DIR}")
        return
    
    images = sorted([f for f in os.listdir(LAB_REPORT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(images)} lab report images")
    
    rows = []
    skipped = 0
    
    for i, img_name in enumerate(images):
        img_path = os.path.join(LAB_REPORT_DIR, img_name)
        
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Processing {i+1}/{len(images)}: {img_name[:60]}...")
        
        # OCR
        ocr_text = ocr_image(img_path)
        if not ocr_text or len(ocr_text) < 20:
            skipped += 1
            continue
        
        # Extract parameters
        params = extract_parameters(ocr_text)
        if len(params) < 1:
            skipped += 1
            continue
        
        # Generate clean summary
        summary = generate_clean_summary(params)
        if not summary:
            skipped += 1
            continue
        
        # Truncate OCR text to 512 tokens worth (~2000 chars) for BART input
        report_text = ocr_text[:2000]
        
        rows.append({
            "report": report_text,
            "summary": summary
        })
    
    print(f"\nDone! Processed: {len(rows)}, Skipped: {skipped}")
    
    if rows:
        # Backup old dataset
        if os.path.exists(OUTPUT_CSV):
            backup = OUTPUT_CSV.replace(".csv", "_backup.csv")
            os.rename(OUTPUT_CSV, backup)
            print(f"Backed up old dataset to {os.path.basename(backup)}")
        
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
        print(f"Saved {len(rows)} training samples to {OUTPUT_CSV}")
        
        # Stats
        avg_report_len = df["report"].str.len().mean()
        avg_summary_len = df["summary"].str.len().mean()
        print(f"Avg report length: {avg_report_len:.0f} chars")
        print(f"Avg summary length: {avg_summary_len:.0f} chars")
    else:
        print("No valid training samples generated!")

if __name__ == "__main__":
    main()
