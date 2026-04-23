"""
Batch translate all glossary terms' definitions to Hindi and Marathi.
Translates the plain-English definitions (not the medical terms) for meaning-based output.
Uses separator-based batching to minimize API calls. Saves progress every 500 terms.
"""
import json, pandas as pd, time, sys
from deep_translator import GoogleTranslator

CORPUS_PATH = "data/medical_corpus.json"
GLOSSARY_PATH = "data/glossary - glossary.csv"
BATCH_SIZE = 20  # definitions per API call (smaller = more reliable)
SEPARATOR = " ||| "
DELAY = 1.0
SAVE_EVERY = 500  # save progress every N terms

# Load existing corpus
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)
existing = set(corpus["medical_terms"].keys())
print(f"Existing corpus terms: {len(existing)}", flush=True)

# Load glossary
glossary = pd.read_csv(GLOSSARY_PATH, engine="python")
to_translate = []
for _, row in glossary.iterrows():
    term = str(row["term"]).strip() if pd.notna(row["term"]) else ""
    simple = str(row["simple"]).strip() if pd.notna(row["simple"]) else ""
    if term and term.lower() not in existing and simple and simple.lower() != "nan":
        to_translate.append((term, simple))

print(f"Terms to translate: {len(to_translate)}", flush=True)

def translate_one(text, target_lang, retries=2):
    """Translate a single text with retries."""
    for attempt in range(retries):
        try:
            t = GoogleTranslator(source="en", target=target_lang)
            result = t.translate(text)
            return result if result else ""
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return ""

def translate_batch_safe(texts, target_lang, lang_name):
    """Translate texts using separator batching with fallback."""
    results = []
    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        combined = SEPARATOR.join(batch)
        try:
            t = GoogleTranslator(source="en", target=target_lang)
            translated = t.translate(combined)
            parts = translated.split("|||")
            parts = [p.strip() for p in parts]
            if len(parts) < len(batch):
                parts.extend([""] * (len(batch) - len(parts)))
            elif len(parts) > len(batch):
                parts = parts[:len(batch)]
            results.extend(parts)
        except Exception as e:
            # Fallback: translate individually
            for txt in batch:
                results.append(translate_one(txt, target_lang))
            time.sleep(2)

        done = min(i + BATCH_SIZE, total)
        batch_num = i // BATCH_SIZE
        if batch_num % 5 == 0:  # Print every 5 batches
            pct = done * 100 // total
            print(f"  {lang_name}: {pct}% ({done}/{total})", flush=True)
        time.sleep(DELAY)
    return results

# Do both languages together per chunk and save periodically
added = 0
chunk_start = 0
CHUNK = SAVE_EVERY

while chunk_start < len(to_translate):
    chunk_end = min(chunk_start + CHUNK, len(to_translate))
    chunk = to_translate[chunk_start:chunk_end]
    defs = [d for _, d in chunk]

    print(f"\n--- Processing terms {chunk_start+1}-{chunk_end} of {len(to_translate)} ---", flush=True)

    print("  Hindi...", flush=True)
    hindi_r = translate_batch_safe(defs, "hi", "Hindi")

    print("  Marathi...", flush=True)
    marathi_r = translate_batch_safe(defs, "mr", "Marathi")

    for j, (term, defn) in enumerate(chunk):
        hi = hindi_r[j] if j < len(hindi_r) else ""
        mr = marathi_r[j] if j < len(marathi_r) else ""
        corpus["medical_terms"][term.lower()] = {
            "english": term,
            "marathi": mr if mr else "",
            "hindi": hi if hi else "",
        }
        added += 1

    # Save progress
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"  Saved! Total corpus now: {len(corpus['medical_terms'])}", flush=True)

    chunk_start = chunk_end

print(f"\nDone! Added {added} terms. Total corpus: {len(corpus['medical_terms'])}", flush=True)
