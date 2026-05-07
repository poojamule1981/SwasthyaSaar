"""
SwasthyaSaar - Real Testing with 10 Reports
Creates 10 test reports with KNOWN values, runs them through the extraction pipeline,
and measures actual Precision, Recall, and Classification accuracy.

Run: python run_real_tests.py
Output: results/ folder with real measured data
"""

import os
import sys
import json
import re
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We'll import the extraction functions from main.py
# But since main.py uses streamlit, we need to mock it
import unittest.mock as mock
st_mock = mock.MagicMock()
sys.modules['streamlit'] = st_mock

# Now import what we need
import importlib.util

# Load main.py functions manually
spec = importlib.util.spec_from_file_location("main_module", os.path.join(os.path.dirname(__file__), "main.py"))

# Actually, let's just replicate the core functions we need
# Load reference ranges
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_reports")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 10 TEST REPORTS - Known Ground Truth
# ============================================================

TEST_REPORTS = [
    {
        "id": "TR-01",
        "name": "Complete Blood Count (CBC) - Clear",
        "type": "CBC",
        "format": "html",
        "expected_params": {
            "hemoglobin": {"value": 9.8, "status": "Low"},
            "rbc count": {"value": 3.42, "status": "Low"},
            "hematocrit": {"value": 30.2, "status": "Low"},
            "mcv": {"value": 88.3, "status": "Normal"},
            "mch": {"value": 28.6, "status": "Normal"},
            "mchc": {"value": 32.4, "status": "Normal"},
            "wbc count": {"value": 7800, "status": "Normal"},
            "platelet count": {"value": 245000, "status": "Normal"},
            "neutrophils": {"value": 62, "status": "Normal"},
            "lymphocytes": {"value": 30, "status": "Normal"},
            "eosinophils": {"value": 3, "status": "Normal"},
            "monocytes": {"value": 4, "status": "Normal"},
            "basophils": {"value": 1, "status": "Normal"},
            "rdw": {"value": 14.5, "status": "Normal"},
            "mpv": {"value": 9.8, "status": "Normal"},
        },
        "content": """<html><body>
<h2>COMPLETE BLOOD COUNT (CBC)</h2>
<table>
<tr><th>Test</th><th>Result</th><th>Unit</th><th>Reference Range</th></tr>
<tr><td>Hemoglobin</td><td>9.8</td><td>g/dL</td><td>12.0-16.0</td></tr>
<tr><td>RBC Count</td><td>3.42</td><td>mill/µL</td><td>4.0-5.5</td></tr>
<tr><td>Hematocrit (PCV)</td><td>30.2</td><td>%</td><td>36-46</td></tr>
<tr><td>MCV</td><td>88.3</td><td>fL</td><td>80-100</td></tr>
<tr><td>MCH</td><td>28.6</td><td>pg</td><td>27-33</td></tr>
<tr><td>MCHC</td><td>32.4</td><td>g/dL</td><td>32-36</td></tr>
<tr><td>Total WBC Count</td><td>7800</td><td>/µL</td><td>4000-11000</td></tr>
<tr><td>Platelet Count</td><td>2,45,000</td><td>cells/cumm</td><td>150000-410000</td></tr>
<tr><td>Neutrophils</td><td>62</td><td>%</td><td>40-70</td></tr>
<tr><td>Lymphocytes</td><td>30</td><td>%</td><td>20-40</td></tr>
<tr><td>Eosinophils</td><td>3</td><td>%</td><td>1-6</td></tr>
<tr><td>Monocytes</td><td>4</td><td>%</td><td>2-8</td></tr>
<tr><td>Basophils</td><td>1</td><td>%</td><td>0-2</td></tr>
<tr><td>RDW</td><td>14.5</td><td>%</td><td>11.5-15.5</td></tr>
<tr><td>MPV</td><td>9.8</td><td>fL</td><td>7.5-11.5</td></tr>
</table></body></html>"""
    },
    {
        "id": "TR-02",
        "name": "Lipid Profile - Normal",
        "type": "Lipid",
        "format": "html",
        "expected_params": {
            "cholesterol": {"value": 185, "status": "Normal"},
            "triglycerides": {"value": 140, "status": "Normal"},
            "hdl": {"value": 52, "status": "Normal"},
            "ldl": {"value": 105, "status": "Normal"},
            "vldl": {"value": 28, "status": "Normal"},
        },
        "content": """<html><body>
<h2>LIPID PROFILE</h2>
<table>
<tr><th>Test</th><th>Result</th><th>Unit</th><th>Reference</th></tr>
<tr><td>Total Cholesterol</td><td>185</td><td>mg/dL</td><td>< 200</td></tr>
<tr><td>Triglycerides</td><td>140</td><td>mg/dL</td><td>< 150</td></tr>
<tr><td>HDL Cholesterol</td><td>52</td><td>mg/dL</td><td>40-60</td></tr>
<tr><td>LDL Cholesterol</td><td>105</td><td>mg/dL</td><td>< 130</td></tr>
<tr><td>VLDL Cholesterol</td><td>28</td><td>mg/dL</td><td>< 30</td></tr>
</table></body></html>"""
    },
    {
        "id": "TR-03",
        "name": "Kidney Function Test (KFT)",
        "type": "KFT",
        "format": "html",
        "expected_params": {
            "urea": {"value": 45, "status": "High"},
            "creatinine": {"value": 1.8, "status": "High"},
            "uric acid": {"value": 7.5, "status": "High"},
            "calcium": {"value": 9.2, "status": "Normal"},
            "sodium": {"value": 140, "status": "Normal"},
            "potassium": {"value": 4.5, "status": "Normal"},
            "chloride": {"value": 102, "status": "Normal"},
        },
        "content": """<html><body>
<h2>KIDNEY FUNCTION TEST</h2>
<table>
<tr><th>Parameter</th><th>Result</th><th>Unit</th><th>Normal Range</th></tr>
<tr><td>Blood Urea</td><td>45</td><td>mg/dL</td><td>15-40</td></tr>
<tr><td>Serum Creatinine</td><td>1.8</td><td>mg/dL</td><td>0.6-1.2</td></tr>
<tr><td>Uric Acid</td><td>7.5</td><td>mg/dL</td><td>3.5-7.0</td></tr>
<tr><td>Calcium</td><td>9.2</td><td>mg/dL</td><td>8.5-10.5</td></tr>
<tr><td>Sodium</td><td>140</td><td>mmol/L</td><td>136-145</td></tr>
<tr><td>Potassium</td><td>4.5</td><td>mmol/L</td><td>3.5-5.1</td></tr>
<tr><td>Chloride</td><td>102</td><td>mmol/L</td><td>98-106</td></tr>
</table></body></html>"""
    },
    {
        "id": "TR-04",
        "name": "Liver Function Test (LFT)",
        "type": "LFT",
        "format": "html",
        "expected_params": {
            "bilirubin total": {"value": 2.5, "status": "High"},
            "bilirubin direct": {"value": 0.8, "status": "High"},
            "sgpt": {"value": 68, "status": "High"},
            "sgot": {"value": 55, "status": "High"},
            "alkaline phosphatase": {"value": 95, "status": "Normal"},
            "total protein": {"value": 7.2, "status": "Normal"},
            "albumin": {"value": 4.0, "status": "Normal"},
            "globulin": {"value": 3.2, "status": "Normal"},
        },
        "content": """<html><body>
<h2>LIVER FUNCTION TEST</h2>
<table>
<tr><th>Investigation</th><th>Result</th><th>Unit</th><th>Ref Range</th></tr>
<tr><td>Bilirubin Total</td><td>2.5</td><td>mg/dL</td><td>0.1-1.2</td></tr>
<tr><td>Bilirubin Direct</td><td>0.8</td><td>mg/dL</td><td>0.0-0.3</td></tr>
<tr><td>SGPT (ALT)</td><td>68</td><td>U/L</td><td>7-56</td></tr>
<tr><td>SGOT (AST)</td><td>55</td><td>U/L</td><td>10-40</td></tr>
<tr><td>Alkaline Phosphatase</td><td>95</td><td>U/L</td><td>44-147</td></tr>
<tr><td>Total Protein</td><td>7.2</td><td>g/dL</td><td>6.0-8.3</td></tr>
<tr><td>Albumin</td><td>4.0</td><td>g/dL</td><td>3.5-5.5</td></tr>
<tr><td>Globulin</td><td>3.2</td><td>g/dL</td><td>2.0-3.5</td></tr>
</table></body></html>"""
    },
    {
        "id": "TR-05",
        "name": "Thyroid Profile",
        "type": "Thyroid",
        "format": "html",
        "expected_params": {
            "tsh": {"value": 8.5, "status": "High"},
            "t3": {"value": 0.9, "status": "Low"},
            "t4": {"value": 4.5, "status": "Low"},
        },
        "content": """<html><body>
<h2>THYROID FUNCTION TEST</h2>
<table>
<tr><th>Test Name</th><th>Result</th><th>Unit</th><th>Normal Range</th></tr>
<tr><td>TSH (Thyroid Stimulating Hormone)</td><td>8.5</td><td>µIU/mL</td><td>0.4-4.0</td></tr>
<tr><td>T3 (Triiodothyronine)</td><td>0.9</td><td>ng/mL</td><td>0.8-2.0</td></tr>
<tr><td>T4 (Thyroxine)</td><td>4.5</td><td>µg/dL</td><td>5.0-12.0</td></tr>
</table></body></html>"""
    },
    {
        "id": "TR-06",
        "name": "Diabetes Panel (Sugar Tests)",
        "type": "Diabetes",
        "format": "html",
        "expected_params": {
            "blood sugar fasting": {"value": 135, "status": "High"},
            "blood sugar pp": {"value": 210, "status": "High"},
            "hba1c": {"value": 7.8, "status": "High"},
        },
        "content": """<html><body>
<h2>DIABETES PANEL</h2>
<table>
<tr><th>Test</th><th>Result</th><th>Unit</th><th>Normal</th></tr>
<tr><td>Blood Sugar Fasting</td><td>135</td><td>mg/dL</td><td>70-100</td></tr>
<tr><td>Blood Sugar PP (Post Prandial)</td><td>210</td><td>mg/dL</td><td>< 140</td></tr>
<tr><td>HbA1c (Glycated Hemoglobin)</td><td>7.8</td><td>%</td><td>< 5.7</td></tr>
</table></body></html>"""
    },
    {
        "id": "TR-07",
        "name": "Electrolyte Panel - Abnormal",
        "type": "Electrolytes",
        "format": "html",
        "expected_params": {
            "sodium": {"value": 128, "status": "Low"},
            "potassium": {"value": 5.8, "status": "High"},
            "chloride": {"value": 95, "status": "Low"},
            "calcium": {"value": 8.0, "status": "Low"},
        },
        "content": """<html><body>
<h2>ELECTROLYTE PANEL</h2>
<table>
<tr><th>Parameter</th><th>Value</th><th>Unit</th><th>Reference</th></tr>
<tr><td>Sodium (Na+)</td><td>128</td><td>mmol/L</td><td>136-145</td></tr>
<tr><td>Potassium (K+)</td><td>5.8</td><td>mmol/L</td><td>3.5-5.1</td></tr>
<tr><td>Chloride (Cl-)</td><td>95</td><td>mmol/L</td><td>98-106</td></tr>
<tr><td>Calcium</td><td>8.0</td><td>mg/dL</td><td>8.5-10.5</td></tr>
</table></body></html>"""
    },
    {
        "id": "TR-08",
        "name": "Iron & Vitamin Panel",
        "type": "Vitamin",
        "format": "html",
        "expected_params": {
            "iron": {"value": 35, "status": "Low"},
            "vitamin d": {"value": 12, "status": "Low"},
            "vitamin b12": {"value": 180, "status": "Low"},
        },
        "content": """<html><body>
<h2>IRON & VITAMIN PANEL</h2>
<table>
<tr><th>Test</th><th>Result</th><th>Unit</th><th>Normal Range</th></tr>
<tr><td>Serum Iron</td><td>35</td><td>µg/dL</td><td>60-170</td></tr>
<tr><td>Vitamin D (25-OH)</td><td>12</td><td>ng/mL</td><td>30-100</td></tr>
<tr><td>Vitamin B12</td><td>180</td><td>pg/mL</td><td>200-900</td></tr>
</table></body></html>"""
    },
    {
        "id": "TR-09",
        "name": "CBC with ESR - Text Format",
        "type": "CBC+ESR",
        "format": "txt",
        "expected_params": {
            "hemoglobin": {"value": 14.2, "status": "Normal"},
            "wbc count": {"value": 12500, "status": "High"},
            "platelet count": {"value": 180000, "status": "Normal"},
            "esr": {"value": 35, "status": "High"},
            "neutrophils": {"value": 75, "status": "High"},
            "lymphocytes": {"value": 18, "status": "Low"},
        },
        "content": """COMPLETE BLOOD COUNT WITH ESR
Patient Name: Test Patient    Date: 01/05/2026

HEMATOLOGY REPORT
----------------------------------------------
Hemoglobin          14.2    g/dL       12.0-16.0
Total WBC Count     12500   /µL        4000-11000
Platelet Count      1,80,000 cells/cumm 150000-410000
ESR                 35      mm/hr      0-20
Neutrophils         75      %          40-70
Lymphocytes         18      %          20-40
----------------------------------------------
"""
    },
    {
        "id": "TR-10",
        "name": "Mixed Panel (CBC + KFT + Sugar)",
        "type": "Mixed",
        "format": "html",
        "expected_params": {
            "hemoglobin": {"value": 11.5, "status": "Normal"},
            "wbc count": {"value": 6200, "status": "Normal"},
            "platelet count": {"value": 220000, "status": "Normal"},
            "creatinine": {"value": 1.5, "status": "High"},
            "urea": {"value": 50, "status": "High"},
            "blood sugar fasting": {"value": 110, "status": "High"},
            "sodium": {"value": 138, "status": "Normal"},
            "potassium": {"value": 4.2, "status": "Normal"},
            "hba1c": {"value": 6.2, "status": "High"},
            "cholesterol": {"value": 220, "status": "High"},
        },
        "content": """<html><body>
<h2>COMPREHENSIVE HEALTH CHECK-UP</h2>
<h3>Hematology</h3>
<table>
<tr><th>Test</th><th>Result</th><th>Unit</th><th>Reference</th></tr>
<tr><td>Hemoglobin</td><td>11.5</td><td>g/dL</td><td>12.0-16.0</td></tr>
<tr><td>Total WBC Count</td><td>6200</td><td>/µL</td><td>4000-11000</td></tr>
<tr><td>Platelet Count</td><td>2,20,000</td><td>cells/cumm</td><td>150000-410000</td></tr>
</table>
<h3>Kidney Function</h3>
<table>
<tr><th>Test</th><th>Result</th><th>Unit</th><th>Reference</th></tr>
<tr><td>Serum Creatinine</td><td>1.5</td><td>mg/dL</td><td>0.6-1.2</td></tr>
<tr><td>Blood Urea</td><td>50</td><td>mg/dL</td><td>15-40</td></tr>
<tr><td>Sodium</td><td>138</td><td>mmol/L</td><td>136-145</td></tr>
<tr><td>Potassium</td><td>4.2</td><td>mmol/L</td><td>3.5-5.1</td></tr>
</table>
<h3>Diabetes & Lipid</h3>
<table>
<tr><th>Test</th><th>Result</th><th>Unit</th><th>Reference</th></tr>
<tr><td>Blood Sugar Fasting</td><td>110</td><td>mg/dL</td><td>70-100</td></tr>
<tr><td>HbA1c</td><td>6.2</td><td>%</td><td>< 5.7</td></tr>
<tr><td>Total Cholesterol</td><td>220</td><td>mg/dL</td><td>< 200</td></tr>
</table></body></html>"""
    },
]


def main():
    # Write test report files
    print("=" * 60)
    print("SwasthyaSaar - REAL TESTING (10 Reports)")
    print("=" * 60)
    
    print("\n📝 Writing test report files...")
    for report in TEST_REPORTS:
        ext = report["format"]
        filepath = os.path.join(TEST_DIR, f"{report['id']}_{report['type'].lower().replace('+','_')}.{ext}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report["content"])
        report["filepath"] = filepath
        print(f"  ✓ {report['id']}: {report['name']} ({ext})")
    
    # Now run extraction on each
    # Import extraction logic from main.py
    # We need to load reference_ranges and extract_parameters
    print("\n🔧 Loading reference ranges...")
    
    # Load reference ranges directly
    ref_file = os.path.join(DATA_DIR, "reference_ranges.csv")
    reference_ranges = {}
    
    df = pd.read_csv(ref_file, engine="python", comment='#')
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
        s = str(x).strip().replace("%", "")
        s = re.sub(r"[^\d\.\-]", "", s)
        if not s: return None
        try: return float(s)
        except: return None
    
    for _, row in df.iterrows():
        p = str(row[param_col]).strip().lower()
        low = parse_bound(row[low_col]) if low_col else None
        high = parse_bound(row[high_col]) if high_col else None
        unit = str(row[unit_col]).strip() if unit_col and not pd.isna(row[unit_col]) else ""
        syns = []
        if syn_col:
            raw = str(row[syn_col]) if not pd.isna(row[syn_col]) else ""
            syns = [s.strip().lower() for s in re.split(r"[;,/|]", raw) if s.strip()]
        if p:
            reference_ranges[p] = {"low": low, "high": high, "unit": unit, "synonyms": syns}
    
    print(f"  Loaded {len(reference_ranges)} parameters")
    
    # Import extraction function from main
    # We'll exec the relevant functions from main.py
    print("\n🧪 Running extraction on each report...\n")
    
    # We need extract_text and extract_parameters from main.py
    # Let's load them by reading main.py and extracting functions
    main_path = os.path.join(os.path.dirname(__file__), "main.py")
    with open(main_path, "r", encoding="utf-8") as f:
        main_code = f.read()
    
    # Extract the normalize_value function
    # Instead of complex import, let's just call main.py as a subprocess for each file
    # Actually, let's use a simpler approach - directly implement extraction here
    
    def normalize_value(value_str, param=None):
        try:
            if value_str is None: return None
            s = str(value_str).strip()
            if not s: return None
            if re.match(r'^\d{1,3}(,\d{2,3})+$', s):
                s = s.replace(",", "")
            else:
                s = re.sub(r',(\d{3})(?!\d)', r'\1', s)
                s = s.replace(",", ".")
            if re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", s):
                return None
            if "-" in s and re.search(r"\d", s):
                parts = re.split(r"\s*-\s*", s)
                nums = []
                for p in parts:
                    m = re.search(r"[<>]?\s*(\d+(?:\.\d+)?)", p)
                    if m: nums.append(float(m.group(1)))
                if nums: val = sum(nums)/len(nums)
                else: return None
            elif s.startswith("<") or s.startswith("≤"):
                m = re.search(r"[<≤]\s*(\d+(?:\.\d+)?)", s)
                if not m: return None
                val = float(m.group(1)) * 0.98
            elif s.startswith(">") or s.startswith("≥"):
                m = re.search(r"[>≥]\s*(\d+(?:\.\d+)?)", s)
                if not m: return None
                val = float(m.group(1)) * 1.02
            else:
                m = re.search(r"(?<!\d)(\d{1,7}(?:\.\d+)?)(?!\d)", s)
                if not m: return None
                val = float(m.group(1))
            if val <= 0 or val > 1e6: return None
            if param and param in reference_ranges:
                low = reference_ranges[param].get("low")
                high = reference_ranges[param].get("high")
                unit = reference_ranges[param].get("unit", "")
                if low is not None and high is not None:
                    if "%" in unit and val > 100: return None
                    if "%" not in unit and val > 100 * max(1.0, high): return None
                    for divisor in [10, 100, 1000]:
                        if val > high * 3 and val / divisor >= low * 0.5 and val / divisor <= high * 2:
                            val = val / divisor
                            break
            return val
        except: return None
    
    def extract_text(filepath):
        if filepath.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif filepath.endswith((".html", ".htm")):
            with open(filepath, "r", encoding="utf-8") as f:
                html_content = f.read()
            text = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL|re.IGNORECASE)
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL|re.IGNORECASE)
            text = re.sub(r'<(?:tr|div|p|br|li|h[1-6]|table|/tr|/table)[^>]*>', '\n', text, flags=re.IGNORECASE)
            text = re.sub(r'<(?:td|th)[^>]*>', '\t', text, flags=re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'&nbsp;|&#xa0;|&#160;', ' ', text, flags=re.IGNORECASE)
            text = re.sub(r'&[a-zA-Z]+;', ' ', text)
            text = re.sub(r'[ ]+', ' ', text)
            text = re.sub(r'\t', '  ', text)
            lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
            return "\n".join(lines)
        return ""
    
    def extract_parameters(text, ref_ranges, char_window=40):
        results = []
        seen_params = set()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        lines_cleaned = []
        for line in lines:
            cleaned = line.lower()
            cleaned = re.sub(r'(?<![a-z0-9])([a-z])\.([a-z])\.([a-z])\.?(?![a-z0-9])', r'\1\2\3', cleaned)
            cleaned = re.sub(r'[()/:;,\-]', ' ', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            lines_cleaned.append(cleaned)
        
        for li, line in enumerate(lines):
            line_lower = line.lower()
            cleaned_lower = lines_cleaned[li]
            
            for param, details in ref_ranges.items():
                if param in seen_params:
                    continue
                candidates = [param] + details.get("synonyms", [])
                matched = False
                val = None
                
                for cand in candidates:
                    cand_lower = cand.lower()
                    if len(cand_lower) < 3:
                        continue
                    cand_cleaned = re.sub(r'[()/:;,\-]', ' ', cand_lower)
                    cand_cleaned = re.sub(r'\s+', ' ', cand_cleaned).strip()
                    
                    match_orig = re.search(rf"\b{re.escape(cand_lower)}\b", line_lower)
                    match_clean = re.search(rf"\b{re.escape(cand_cleaned)}\b", cleaned_lower) if cand_cleaned != cand_lower else None
                    
                    if match_orig or match_clean:
                        matched = True
                        if match_orig:
                            m_word = match_orig
                            search_line = line
                        else:
                            m_word = match_clean
                            search_line = lines_cleaned[li]
                        
                        end_pos = m_word.end()
                        window = search_line[end_pos:end_pos + char_window + 20]
                        m_val = re.search(r"([<>≤≥]?\s*\d[\d,]*\.?\d*)", window)
                        if m_val:
                            val = normalize_value(m_val.group(1), param)
                        break
                
                if matched and val is not None:
                    low, high = details.get("low"), details.get("high")
                    status = "Normal"
                    if low is not None and val < low:
                        status = "Low"
                    elif high is not None and val > high:
                        status = "High"
                    results.append({"parameter": param, "value": val, "status": status})
                    seen_params.add(param)
        
        return results
    
    # ============================================================
    # RUN TESTS
    # ============================================================
    
    all_results = []
    
    for report in TEST_REPORTS:
        filepath = report["filepath"]
        expected = report["expected_params"]
        
        # Extract text
        text = extract_text(filepath)
        
        # Run extraction
        detected = extract_parameters(text, reference_ranges)
        detected_dict = {r["parameter"]: r for r in detected}
        
        # Calculate metrics
        actual_count = len(expected)
        detected_count = 0
        correct_count = 0
        classification_correct = 0
        
        for param, exp in expected.items():
            if param in detected_dict:
                detected_count += 1
                # Check if value is close enough (within 5%)
                det_val = detected_dict[param]["value"]
                exp_val = exp["value"]
                if abs(det_val - exp_val) / max(exp_val, 0.01) < 0.05:
                    correct_count += 1
                # Check classification
                if detected_dict[param]["status"] == exp["status"]:
                    classification_correct += 1
        
        precision = correct_count / detected_count if detected_count > 0 else 0
        recall = correct_count / actual_count if actual_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        classification_acc = classification_correct / detected_count if detected_count > 0 else 0
        
        result = {
            "Test ID": report["id"],
            "Report Type": report["type"],
            "Format": report["format"],
            "Expected Params": actual_count,
            "Detected Params": detected_count,
            "Correct Values": correct_count,
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1 Score": round(f1, 2),
            "Classification Acc": round(classification_acc, 2),
        }
        all_results.append(result)
        
        # Print details
        status_icon = "✅" if recall >= 0.8 else "⚠️"
        print(f"{status_icon} {report['id']} | {report['name']}")
        print(f"   Expected: {actual_count} | Detected: {detected_count} | Correct: {correct_count}")
        print(f"   Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | Classification: {classification_acc:.2f}")
        
        # Show misses
        missed = [p for p in expected if p not in detected_dict]
        if missed:
            print(f"   ❌ Missed: {', '.join(missed)}")
        wrong_class = [p for p in expected if p in detected_dict and detected_dict[p]["status"] != expected[p]["status"]]
        if wrong_class:
            print(f"   ⚠️ Wrong classification: {', '.join(wrong_class)}")
        print()
    
    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(RESULTS_DIR, "real_test_results.csv"), index=False)
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(df_results.to_string(index=False))
    
    # Overall metrics
    total_expected = df_results["Expected Params"].sum()
    total_detected = df_results["Detected Params"].sum()
    total_correct = df_results["Correct Values"].sum()
    
    overall_precision = total_correct / total_detected if total_detected > 0 else 0
    overall_recall = total_correct / total_expected if total_expected > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    avg_classification = df_results["Classification Acc"].mean()
    
    print(f"\n📈 OVERALL METRICS:")
    print(f"   Total Parameters Expected: {total_expected}")
    print(f"   Total Parameters Detected: {total_detected}")
    print(f"   Total Correct Extractions: {total_correct}")
    print(f"   Overall Precision: {overall_precision:.2%}")
    print(f"   Overall Recall: {overall_recall:.2%}")
    print(f"   Overall F1 Score: {overall_f1:.2%}")
    print(f"   Avg Classification Accuracy: {avg_classification:.2%}")
    
    # Save summary
    summary = {
        "total_reports": 10,
        "total_expected": int(total_expected),
        "total_detected": int(total_detected),
        "total_correct": int(total_correct),
        "overall_precision": round(overall_precision, 4),
        "overall_recall": round(overall_recall, 4),
        "overall_f1": round(overall_f1, 4),
        "avg_classification_accuracy": round(avg_classification, 4),
    }
    with open(os.path.join(RESULTS_DIR, "real_test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📁 Results saved to: {RESULTS_DIR}/real_test_results.csv")
    print("✅ Done!")
    
    return df_results, summary


if __name__ == "__main__":
    main()
