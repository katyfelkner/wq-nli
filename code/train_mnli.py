from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import sys
import numpy as np

# ARGUMENTS
# sys.argv[1] pretrained model path
# sys.argv[2] custom save location for finetuned model (optional)

# tokenize function
def tokenize_function(examples):
    if tokenizer.pad_token:
        return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=model.config.max_position_embeddings)
    else:
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(examples["premise"], examples["hypothesis"], padding=True, truncation=True, max_length=model.config.max_position_embeddings)
    
# eval function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# load dataset
dataset = load_dataset("nyu-mll/multi_nli")
metric = evaluate.load("accuracy")

# load pretrained model
pretrained_model_path = sys.argv[1] # path to pretrained model
# fairly hacky handling of filenames - could fix by reading config file instead of hard coding for my file structure
# deal with trailing slash
if pretrained_model_path[-1] == '/': pretrained_model_path = pretrained_model_path[:-1]
base_model_path = "../../new_finetune/pretrained/" + pretrained_model_path.split('/')[-1].split("-finetuned")[0]
model_name = pretrained_model_path.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
print("load model...")
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=3)

print("begin tokenizing...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# training arguments
training_args = TrainingArguments( 
    f"{model_name}-finetuned-mnli",
    evaluation_strategy = "steps",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    num_train_epochs = 4.0,
    fp16=True,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 10 # for speed
)

# fine tune model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
    compute_metrics=compute_metrics,
)

print("Begin finetuning...")
trainer.train()

if len(sys.argv) > 2: # user supplied save location
    model.save_pretrained(sys.argv[2])
else:
    # construct a sensible name for a subdirectory of pwd
    model.save_pretrained(f"{model_name}-finetuned-mnli")
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import sys
import numpy as np

# ARGUMENTS
# sys.argv[1] pretrained model path
# sys.argv[2] custom save location for finetuned model (optional)

# tokenize function
def tokenize_function(examples):
    if tokenizer.pad_token:
        return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=model.config.max_position_embeddings)
    else:
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(examples["premise"], examples["hypothesis"], padding=True, truncation=True, max_length=model.config.max_position_embeddings)
    
# eval function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# load dataset
dataset = load_dataset("nyu-mll/multi_nli")
metric = evaluate.load("accuracy")

# load pretrained model
pretrained_model_path = sys.argv[1] # path to pretrained model
# fairly hacky handling of filenames - could fix by reading config file instead of hard coding for my file structure
# deal with trailing slash
if pretrained_model_path[-1] == '/': pretrained_model_path = pretrained_model_path[:-1]
base_model_path = "../../new_finetune/pretrained/" + pretrained_model_path.split('/')[-1].split("-finetuned")[0]
model_name = pretrained_model_path.split('/')[-1]
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
print("load model...")
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=3)

print("begin tokenizing...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# training arguments
training_args = TrainingArguments( 
    f"{model_name}-finetuned-mnli",
    evaluation_strategy = "steps",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    num_train_epochs = 4.0,
    fp16=True,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 10 # for speed
)

# fine tune model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
    compute_metrics=compute_metrics,
)

print("Begin finetuning...")
trainer.train()

if len(sys.argv) > 2: # user supplied save location
    model.save_pretrained(sys.argv[2])
else:
    # construct a sensible name for a subdirectory of pwd
    model.save_pretrained(f"{model_name}-finetuned-mnli")
