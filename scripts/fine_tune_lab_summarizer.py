import os
import numpy as np
import re
import pandas as pd
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import cv2
from transformers import pipeline

# -------------------------------
# Config
# -------------------------------
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODELS_DIR)
REPORTS_DIR = os.path.join(BASE_DIR, "data", "lab_report")  # folder with your lab reports
POPPLER_PATH = os.path.join(os.environ.get("LOCALAPPDATA", ""), "poppler", "poppler-24.08.0", "Library", "bin")
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OUTPUT_CSV = os.path.join(BASE_DIR, "lab_reports_dataset.csv")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# -------------------------------
# OCR Functions
# -------------------------------
def preprocess_image(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(gray)

def extract_text(file_path):
    text = ""
    try:
        if file_path.lower().endswith(".pdf"):
            pages = convert_from_path(file_path, dpi=400, poppler_path=POPPLER_PATH)
            for page in pages:
                processed = preprocess_image(page)
                text += pytesseract.image_to_string(processed, config="--psm 6") + "\n"
        else:
            img = Image.open(file_path)
            processed = preprocess_image(img)
            text += pytesseract.image_to_string(processed, config="--psm 6")
    except Exception as e:
        print(f"❌ OCR failed for {file_path}: {e}")
    return re.sub(r"\s+", " ", text).strip()

# -------------------------------
# Summarization Pipeline
# -------------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text):
    if not text.strip():
        return ""
    try:
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"❌ Summarization failed: {e}")
        return ""

# -------------------------------
# Load existing CSV
# -------------------------------
if os.path.exists(OUTPUT_CSV):
    df = pd.read_csv(OUTPUT_CSV)
    existing_texts = set(df['report'].astype(str))
    print(f"✅ CSV already exists: {OUTPUT_CSV}, {len(existing_texts)} reports found.")
else:
    df = pd.DataFrame(columns=["report", "summary"])
    existing_texts = set()
    print("ℹ️ CSV does not exist, starting fresh.")

# -------------------------------
# Main: Create Dataset
# -------------------------------
dataset = []

files = [f for f in os.listdir(REPORTS_DIR) if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.txt'))]

for i, f in enumerate(files, 1):
    path = os.path.join(REPORTS_DIR, f)
    print(f"[{i}/{len(files)}] Processing: {f}")

    # Extract text
    text = extract_text(path)
    if not text:
        print(f"⚠️ No text extracted from {f}, skipping.")
        continue

    # Skip if text already exists in CSV
    if text in existing_texts:
        print(f"ℹ️ Already processed, skipping {f}.")
        continue

    # Generate summary
    summary = generate_summary(text)
    if not summary:
        print(f"⚠️ No summary generated for {f}, skipping.")
        continue

    dataset.append({"report": text, "summary": summary})

# -------------------------------
# Save to CSV
# -------------------------------
if dataset:
    df = pd.concat([df, pd.DataFrame(dataset)], ignore_index=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Dataset saved: {OUTPUT_CSV}")
else:
    print("ℹ️ No new reports to process.")
