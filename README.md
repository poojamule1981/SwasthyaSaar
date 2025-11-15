# 🩺 SwasthyaSaar  
### AI-powered Medical Report Understanding System  
**Simple explanations of complex medical reports — for everyone.**

---

## 📌 Overview  
**SwasthyaSaar** is an AI-driven application that extracts text from medical reports, identifies complex medical jargon, and provides **easy-to-understand summaries** in simple language (English / Hindi / Marathi support).

This tool helps patients and families quickly understand medical terms, lab values, and report findings using OCR + NLP + Translation.

---

## 🚀 Features  

### 🔍 **1. OCR (Image/PDF → Text)**
- Extracts text from scanned medical reports  
- Uses **Tesseract OCR**  
- Supports JPG, PNG, PDF (scanned), etc.

### 🩻 **2. Medical Term Detection**
- Detects complex medical terms  
- Searches in custom-built medical glossary  
- Uses fuzzy matching (RapidFuzz)

### 📝 **3. Easy Explanation Generator**
- Converts difficult terms into simple-language definitions  
- Provides short, understandable summaries  
- Supports multilingual output

### 🌐 **4. Translation**
- Simple-language explanation translated using **googletrans**  
- Languages supported:
  - English
  - Hindi  
  - Marathi  

### 🧠 **5. AI Model (Optional)**
- Custom fine-tunable summarization model  
- Folder included for future training (`models/`)

### 🧵 **6. Clean Streamlit UI**
- Beautiful interface  
- Upload → Analyze → Understand  
- Error-handling and clean formatting

---

## 📂 Folder Structure  
SwasthyaSaar/
│── main.py
│── requirements.txt
│── medical_jargon.json
│── create_jargon_json.py
│── fine_tune_lab_reports.py
│── lab_reports_dataset.csv
│── data/
│ ├── glossary.csv
│
└── models/
├── fine_tune_lab_summarizer.py
├── lab_summarizer/
└── trained_lab_summarizer/ 



---

## 🔧 Installation & Setup  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/poojamule1981/SwasthyaSaar.git
cd SwasthyaSaar
2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Install Tesseract OCR

Windows users download from:
https://github.com/UB-Mannheim/tesseract/wiki

Then add to PATH.

4️⃣ Run the application
streamlit run main.py
