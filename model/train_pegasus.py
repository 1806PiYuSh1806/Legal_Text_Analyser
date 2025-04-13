import json
import torch
from datasets import Dataset
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Load data from JSON
def load_data(file_path):
    with open(file_path, 'r') as f:
        raw_data = json.load(f)
    data = [{'text': item['text'], 'summary': item['summary']} for item in raw_data]
    return Dataset.from_list(data)

# Load dataset
train_dataset = load_data("ca_test_data_final_OFFICIAL.json")
val_dataset = load_data("ca_test_data_final_OFFICIAL.json")  # use different set if available

# Load tokenizer and model
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Preprocessing
def preprocess(example):
    input = tokenizer(example['text'], truncation=True, padding="max_length", max_length=1024)
    target = tokenizer(example['summary'], truncation=True, padding="max_length", max_length=128)
    input['labels'] = target['input_ids']
    return input

# Apply preprocessing
train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=["text", "summary"])
val_dataset = val_dataset.map(preprocess, batched=True, remove_columns=["text", "summary"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./pegasus_billsum",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save model
model.save_pretrained("./pegasus_billsum")
tokenizer.save_pretrained("./pegasus_billsum")
