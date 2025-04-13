# train_bart.py
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json

# Load custom dataset
with open("ca_test_data_final_OFFICIAL.json", "r") as f:
    data = json.load(f)

# Convert to Hugging Face dataset
dataset = Dataset.from_list(data)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

# Preprocessing function
def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], max_length=1024, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess_function, batched=True)

# Training args
training_args = TrainingArguments(
    output_dir="./results_bart",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs_bart",
    logging_steps=10,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./bart-billsum")
tokenizer.save_pretrained("./bart-billsum")


# train_pegasus.py
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json

# Load custom dataset
with open("ca_test_data_final_OFFICIAL.json", "r") as f:
    data = json.load(f)

# Convert to Hugging Face dataset
dataset = Dataset.from_list(data)

tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")

# Preprocessing function
def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], max_length=1024, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess_function, batched=True)

# Training args
training_args = TrainingArguments(
    output_dir="./results_pegasus",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs_pegasus",
    logging_steps=10,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./pegasus-billsum")
tokenizer.save_pretrained("./pegasus-billsum")
