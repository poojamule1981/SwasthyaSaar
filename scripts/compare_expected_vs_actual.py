"""
SwasthyaSaar - EXPECTED vs ACTUAL Comparison
Shows side-by-side: What we KNOW the value is vs What the SYSTEM extracted.
Like: "Report says Hemoglobin = 9.8, System detected = 9.8 → MATCH ✅"

For blackbook Chapter 6/7 - real analysis with 10 reports.
"""

import os
import sys
import json
import re
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEST_DIR = os.path.join(PROJECT_ROOT, "test_reports")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load reference ranges
ref_file = os.path.join(DATA_DIR, "reference_ranges.csv")
reference_ranges = {}
df_ref = pd.read_csv(ref_file, engine="python", comment='#')

param_col = None
low_col = None
high_col = None
unit_col = None
syn_col = None
for c in df_ref.columns:
    lc = c.lower().strip()
    if lc in ("parameter", "param", "name"): param_col = c
    if lc in ("low", "min", "lbound"): low_col = c
    if lc in ("high", "max", "ubound"): high_col = c
    if lc in ("unit", "units"): unit_col = c
    if lc in ("synonyms", "synonym"): syn_col = c
if param_col is None: param_col = df_ref.columns[0]

def parse_bound(x):
    if pd.isna(x): return None
    s = str(x).strip().replace("%", "")
    s = re.sub(r"[^\d\.\-]", "", s)
    if not s: return None
    try: return float(s)
    except: return None

for _, row in df_ref.iterrows():
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

# ---- Extraction functions (same as run_real_tests.py) ----
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
        if re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", s): return None
        if "-" in s and re.search(r"\d", s):
            parts = re.split(r"\s*-\s*", s)
            nums = []
            for p in parts:
                m = re.search(r"[<>]?\s*(\d+(?:\.\d+)?)", p)
                if m: nums.append(float(m.group(1)))
            if nums: val = sum(nums)/len(nums)
            else: return None
        elif s.startswith("<") or s.startswith("\u2264"):
            m = re.search(r"[<\u2264]\s*(\d+(?:\.\d+)?)", s)
            if not m: return None
            val = float(m.group(1)) * 0.98
        elif s.startswith(">") or s.startswith("\u2265"):
            m = re.search(r"[>\u2265]\s*(\d+(?:\.\d+)?)", s)
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
            if param in seen_params: continue
            candidates = [param] + details.get("synonyms", [])
            matched = False
            val = None
            for cand in candidates:
                cand_lower = cand.lower()
                if len(cand_lower) < 3: continue
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
                    m_val = re.search(r"([<>\u2264\u2265]?\s*\d[\d,]*\.?\d*)", window)
                    if m_val:
                        val = normalize_value(m_val.group(1), param)
                    break
            if matched and val is not None:
                low, high = details.get("low"), details.get("high")
                status = "Normal"
                if low is not None and val < low: status = "Low"
                elif high is not None and val > high: status = "High"
                results.append({"parameter": param, "value": val, "status": status})
                seen_params.add(param)
    return results

# ============================================================
# TEST REPORTS DATA (same as run_real_tests.py)
# ============================================================
TEST_REPORTS = [
    {"id": "TR-01", "name": "CBC (Clear HTML)", "type": "CBC", "format": "html",
     "expected": {"hemoglobin": (9.8, "Low"), "rbc count": (3.42, "Low"), "hematocrit": (30.2, "Low"),
                  "mcv": (88.3, "Normal"), "mch": (28.6, "Normal"), "mchc": (32.4, "Normal"),
                  "wbc count": (7800, "Normal"), "platelet count": (245000, "Normal"),
                  "neutrophils": (62, "Normal"), "lymphocytes": (30, "Normal"),
                  "eosinophils": (3, "Normal"), "monocytes": (4, "Normal"),
                  "basophils": (1, "Normal"), "rdw": (14.5, "Normal"), "mpv": (9.8, "Normal")}},
    {"id": "TR-02", "name": "Lipid Profile", "type": "Lipid", "format": "html",
     "expected": {"cholesterol": (185, "Normal"), "triglycerides": (140, "Normal"),
                  "hdl": (52, "Normal"), "ldl": (105, "Normal"), "vldl": (28, "Normal")}},
    {"id": "TR-03", "name": "Kidney Function (KFT)", "type": "KFT", "format": "html",
     "expected": {"urea": (45, "High"), "creatinine": (1.8, "High"), "uric acid": (7.5, "High"),
                  "calcium": (9.2, "Normal"), "sodium": (140, "Normal"),
                  "potassium": (4.5, "Normal"), "chloride": (102, "Normal")}},
    {"id": "TR-04", "name": "Liver Function (LFT)", "type": "LFT", "format": "html",
     "expected": {"bilirubin total": (2.5, "High"), "bilirubin direct": (0.8, "High"),
                  "sgpt": (68, "High"), "sgot": (55, "High"),
                  "alkaline phosphatase": (95, "Normal"), "total protein": (7.2, "Normal"),
                  "albumin": (4.0, "Normal"), "globulin": (3.2, "Normal")}},
    {"id": "TR-05", "name": "Thyroid Profile", "type": "Thyroid", "format": "html",
     "expected": {"tsh": (8.5, "High"), "t3": (0.9, "Low"), "t4": (4.5, "Low")}},
    {"id": "TR-06", "name": "Diabetes Panel", "type": "Diabetes", "format": "html",
     "expected": {"blood sugar fasting": (135, "High"), "blood sugar pp": (210, "High"),
                  "hba1c": (7.8, "High")}},
    {"id": "TR-07", "name": "Electrolyte Panel", "type": "Electrolytes", "format": "html",
     "expected": {"sodium": (128, "Low"), "potassium": (5.8, "High"),
                  "chloride": (95, "Low"), "calcium": (8.0, "Low")}},
    {"id": "TR-08", "name": "Iron & Vitamin Panel", "type": "Vitamin", "format": "html",
     "expected": {"iron": (35, "Low"), "vitamin d": (12, "Low"), "vitamin b12": (180, "Low")}},
    {"id": "TR-09", "name": "CBC + ESR (Text)", "type": "CBC+ESR", "format": "txt",
     "expected": {"hemoglobin": (14.2, "Normal"), "wbc count": (12500, "High"),
                  "platelet count": (180000, "Normal"), "esr": (35, "High"),
                  "neutrophils": (75, "High"), "lymphocytes": (18, "Low")}},
    {"id": "TR-10", "name": "Mixed Panel", "type": "Mixed", "format": "html",
     "expected": {"hemoglobin": (11.5, "Normal"), "wbc count": (6200, "Normal"),
                  "platelet count": (220000, "Normal"), "creatinine": (1.5, "High"),
                  "urea": (50, "High"), "blood sugar fasting": (110, "High"),
                  "sodium": (138, "Normal"), "potassium": (4.2, "Normal"),
                  "hba1c": (6.2, "High"), "cholesterol": (220, "High")}},
]


def main():
    print("=" * 80)
    print("SwasthyaSaar — EXPECTED vs ACTUAL COMPARISON")
    print("What value is in the report vs What the system detected")
    print("=" * 80)

    all_rows = []

    for report in TEST_REPORTS:
        ext = report["format"]
        filepath = os.path.join(TEST_DIR, f"{report['id']}_{report['type'].lower().replace('+','_')}.{ext}")
        
        if not os.path.exists(filepath):
            print(f"\n⚠️ File not found: {filepath} — Run run_real_tests.py first!")
            continue

        text = extract_text(filepath)
        detected = extract_parameters(text, reference_ranges)
        detected_dict = {r["parameter"]: r for r in detected}

        print(f"\n{'─' * 80}")
        print(f"📋 {report['id']}: {report['name']} ({report['format'].upper()})")
        print(f"{'─' * 80}")
        print(f"{'Parameter':<22} {'Expected Value':<16} {'System Value':<16} {'Value Match':<12} {'Expected Status':<16} {'System Status':<14} {'Status Match'}")
        print(f"{'─'*22} {'─'*15} {'─'*15} {'─'*11} {'─'*15} {'─'*13} {'─'*12}")

        for param, (exp_val, exp_status) in report["expected"].items():
            if param in detected_dict:
                sys_val = detected_dict[param]["value"]
                sys_status = detected_dict[param]["status"]
                
                # Value match check (within 5%)
                if abs(sys_val - exp_val) / max(exp_val, 0.01) < 0.05:
                    val_match = "✅ YES"
                    val_match_bool = True
                else:
                    val_match = f"❌ NO"
                    val_match_bool = False
                
                # Status match
                if sys_status == exp_status:
                    stat_match = "✅ YES"
                    stat_match_bool = True
                else:
                    stat_match = f"❌ NO"
                    stat_match_bool = False
                
                sys_val_str = f"{sys_val:g}"
            else:
                sys_val = None
                sys_val_str = "NOT FOUND"
                sys_status = "—"
                val_match = "❌ MISSED"
                stat_match = "❌ MISSED"
                val_match_bool = False
                stat_match_bool = False
            
            print(f"{param:<22} {str(exp_val):<16} {sys_val_str:<16} {val_match:<12} {exp_status:<16} {sys_status:<14} {stat_match}")
            
            all_rows.append({
                "Report": report["id"],
                "Report Type": report["type"],
                "Parameter": param,
                "Expected Value": exp_val,
                "System Value": sys_val if sys_val else "NOT FOUND",
                "Value Match": val_match_bool,
                "Expected Status": exp_status,
                "System Status": sys_status,
                "Status Match": stat_match_bool,
            })

    # Save detailed comparison as CSV
    df_compare = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "expected_vs_actual_comparison.csv")
    df_compare.to_csv(csv_path, index=False)

    # ============================================================
    # OVERALL SUMMARY
    # ============================================================
    total = len(all_rows)
    val_matches = sum(1 for r in all_rows if r["Value Match"])
    stat_matches = sum(1 for r in all_rows if r["Status Match"])
    missed = sum(1 for r in all_rows if r["System Value"] == "NOT FOUND")
    detected_total = total - missed

    print(f"\n{'=' * 80}")
    print(f"📊 OVERALL COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(f"   Total Parameters Tested:    {total}")
    print(f"   System Detected:            {detected_total}/{total} ({detected_total/total*100:.1f}%)")
    print(f"   Value Correctly Matched:    {val_matches}/{total} ({val_matches/total*100:.1f}%)")
    print(f"   Status Correctly Matched:   {stat_matches}/{total} ({stat_matches/total*100:.1f}%)")
    print(f"   Missed (Not Detected):      {missed}/{total} ({missed/total*100:.1f}%)")
    
    # Show all mismatches
    wrong_vals = [r for r in all_rows if r["System Value"] != "NOT FOUND" and not r["Value Match"]]
    missed_params = [r for r in all_rows if r["System Value"] == "NOT FOUND"]
    wrong_status = [r for r in all_rows if r["System Value"] != "NOT FOUND" and not r["Status Match"]]
    
    if wrong_vals:
        print(f"\n⚠️ WRONG VALUES ({len(wrong_vals)}):")
        for r in wrong_vals:
            print(f"   • {r['Report']} | {r['Parameter']}: Expected={r['Expected Value']}, Got={r['System Value']}")
    
    if missed_params:
        print(f"\n❌ MISSED PARAMETERS ({len(missed_params)}):")
        for r in missed_params:
            print(f"   • {r['Report']} | {r['Parameter']}: Expected={r['Expected Value']}, System=NOT FOUND")
    
    if wrong_status:
        print(f"\n⚠️ WRONG STATUS ({len(wrong_status)}):")
        for r in wrong_status:
            print(f"   • {r['Report']} | {r['Parameter']}: Expected={r['Expected Status']}, Got={r['System Status']}")
    
    print(f"\n📁 Detailed CSV saved: {csv_path}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
