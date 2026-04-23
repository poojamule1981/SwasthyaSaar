# 🩺 SwasthyaSaar

### An NLP-Based Solution for Simplifying and Summarizing Medical Reports

**Making complex medical reports easy to understand — for patients, families, and caregivers.**

---

## 📌 Overview

**SwasthyaSaar** is an AI-powered web application that takes scanned or digital medical lab reports and converts them into **simple, patient-friendly summaries**. It extracts parameters from the report, compares them against standard reference ranges, flags abnormal values, and explains each term in plain language.

The app supports **English, Hindi, and Marathi** output — making healthcare information accessible to non-English speakers across India.

### What It Does

1. **Upload** a lab report (PDF, image, or text file)
2. **Extracts** text using OCR (Tesseract)
3. **Detects** the report type (CBC, LFT, KFT, Thyroid, etc.)
4. **Identifies** medical parameters and their values
5. **Compares** values against biological reference ranges
6. **Flags** abnormal results (High / Low / Normal)
7. **Explains** each parameter in simple language with patient-friendly meanings
8. **Summarizes** the overall report using a fine-tuned BART model
9. **Translates** the output into Hindi or Marathi (meaning-based, not transliteration)

---

## 🚀 Features

### 🔍 OCR & Text Extraction
- Extracts text from scanned medical reports using **Tesseract OCR**
- Supports PDF (scanned & digital), JPG, PNG, and TXT files
- Uses **pdf2image** + **Poppler** for PDF page rendering
- Preprocessing with **OpenCV** for better OCR accuracy

### 🧪 Report Type Detection
- Automatically detects the type of lab report from extracted text
- Supports **23 report types**: CBC, LFT, KFT, Lipid Profile, Thyroid, Diabetes, Electrolytes, Iron Studies, Urine Routine, Coagulation, CRP/Inflammatory Markers, Cardiac Markers, Hormones, Hepatitis, ABG, Vitamins & Minerals, Pancreatic Enzymes, Tumor Markers, Autoimmune Markers, Stool Examination, Semen Analysis, CSF Analysis, Allergy Panel

### 📊 Parameter Extraction & Analysis
- Extracts **239 medical parameters** with their values and units
- Compares against **biological reference ranges** (age/gender-aware)
- Flags each parameter as **Normal**, **High**, or **Low**
- Handles edge cases: percentage values, ranges, non-numeric entries

### 📝 Patient-Friendly Explanations
- **26,460 medical term definitions** in the glossary
- Each parameter gets a simple English explanation
- Uses fuzzy matching (**RapidFuzz**) to match OCR-noisy terms
- Multi-source lookup: glossary → medical corpus → jargon dictionary

### 🌐 Multilingual Support (English / Hindi / Marathi)
- **Meaning-based translations** (not transliteration)
- Hand-crafted Hindi & Marathi translations for all 239 reference range parameters
- Additional translations via **medical_corpus.json** for glossary terms
- Fallback to Google Translate API for uncovered terms

### 🧠 AI Summarization
- Fine-tuned **BART** (facebook/bart-large-cnn) model for medical report summarization
- Generates a concise overall summary of the report findings
- Model stored locally in `models/lab_summarizer/`

### 🧵 Streamlit Web Interface
- Clean, professional UI with upload → analyze → download workflow
- Downloadable patient report (text format)
- Language selector (English / Hindi / Marathi)
- Error handling with informative messages

---

## 📂 Folder Structure

```
SwasthyaSaar/
├── main.py                            # Streamlit app (entry point)
├── requirements.txt                   # Python dependencies
├── README.md
├── LICENSE
│
├── data/                              # Runtime data files
│   ├── glossary - glossary.csv        # 26,460 medical term definitions
│   ├── reference_ranges.csv           # 239 parameters, 23 report sections
│   ├── medical_corpus.json            # Hindi/Marathi translations
│   └── medical_jargon.json            # 42,693 term fallback dictionary
│
├── models/                            # ML models
│   └── lab_summarizer/                # Fine-tuned BART summarization model
│
└── scripts/                           # One-time utility & training scripts
    ├── create_jargon_json.py          # Generates medical_jargon.json
    ├── fine_tune_lab_reports.py        # BART model fine-tuning script
    ├── fine_tune_lab_summarizer.py     # Training data generator
    ├── lab_reports_dataset.csv         # Training dataset (report/summary pairs)
    └── translate_glossary_to_corpus.py # Batch Hindi/Marathi translation utility
```

---

## 🔧 Installation & Setup

### Prerequisites
- **Python 3.10+**
- **Tesseract OCR** installed and added to PATH
- **Poppler** (for PDF rendering)

### 1. Clone the repository
```bash
git clone https://github.com/poojamule1981/SwasthyaSaar.git
cd SwasthyaSaar
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR
- **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

### 5. Install Poppler (for PDF support)
- **Windows**: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases) and add `bin/` to PATH
- **Linux**: `sudo apt install poppler-utils`
- **macOS**: `brew install poppler`

### 6. Run the application
```bash
streamlit run main.py
```
The app will open at `http://localhost:8501`

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Web UI | Streamlit |
| OCR Engine | Tesseract OCR + pytesseract |
| PDF Rendering | pdf2image + Poppler |
| Image Processing | OpenCV, Pillow |
| Text Matching | RapidFuzz (fuzzy matching) |
| NLP / Regex | Python regex, pandas |
| Summarization | Fine-tuned BART (HuggingFace Transformers) |
| Translation | Google Translate (googletrans) + offline corpus |
| Data Format | CSV, JSON |

---

## 📋 Supported Report Types

| # | Report Type | # | Report Type |
|---|---|---|---|
| 1 | Complete Blood Count (CBC) | 13 | Hormones |
| 2 | Liver Function Test (LFT) | 14 | Hepatitis Panel |
| 3 | Kidney Function Test (KFT) | 15 | Arterial Blood Gas (ABG) |
| 4 | Lipid Profile | 16 | Vitamins & Minerals |
| 5 | Thyroid Function Test | 17 | Pancreatic Enzymes |
| 6 | Diabetes (HbA1c, FBS, RBS) | 18 | Tumor Markers |
| 7 | Electrolytes | 19 | Autoimmune Markers |
| 8 | Iron Studies | 20 | Stool Examination |
| 9 | Urine Routine | 21 | Semen Analysis |
| 10 | Coagulation (PT/INR, APTT) | 22 | CSF Analysis |
| 11 | Inflammatory Markers (CRP, ESR) | 23 | Allergy Panel |
| 12 | Cardiac Markers | | |

---

## 📸 How to Use

1. Open the app in your browser (`streamlit run main.py`)
2. Select your preferred language (English / Hindi / Marathi)
3. Upload a lab report file (PDF, JPG, PNG, or TXT)
4. Click **Analyze**
5. View the patient-friendly report with:
   - Extracted parameters with Normal/High/Low status
   - Simple explanations for each medical term
   - AI-generated overall summary
6. Download the report as a text file

---

## 👥 Authors

- **Pooja Mule** — [GitHub](https://github.com/poojamule1981)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
