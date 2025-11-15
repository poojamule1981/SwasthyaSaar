import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

# ===== 1. Load CSV =====
csv_path = r"C:\Users\Pooja\nlp project\lab_reports_dataset.csv"
df = pd.read_csv(csv_path)
print("Columns:", df.columns)

# ===== 2. Convert to HuggingFace Dataset =====
dataset = Dataset.from_pandas(df)

# Split into train and eval
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ===== 3. Load Tokenizer and Model =====
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ===== 4. Tokenization function =====
def tokenize_function(example):
    # tokenize inputs
    inputs = tokenizer(example["report"], truncation=True, padding="max_length", max_length=512)
    # tokenize targets (summary)
    targets = tokenizer(example["summary"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Map tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# ===== 5. Remove original columns safely =====
columns_to_remove = [col for col in ["report", "summary", "__index_level_0__"] if col in train_dataset.column_names]
train_dataset = train_dataset.remove_columns(columns_to_remove)

columns_to_remove_eval = [col for col in ["report", "summary", "__index_level_0__"] if col in eval_dataset.column_names]
eval_dataset = eval_dataset.remove_columns(columns_to_remove_eval)

# ===== 6. Training Arguments =====
training_args = TrainingArguments(
    output_dir="./models/lab_summarizer",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    eval_strategy="steps",  # <-- changed from evaluation_strategy
    eval_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    report_to="none",
)


# ===== 7. Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# ===== 8. Train =====
trainer.train()

# ===== 9. Save Model =====
trainer.save_model("./models/lab_summarizer")
tokenizer.save_pretrained("./models/lab_summarizer")
