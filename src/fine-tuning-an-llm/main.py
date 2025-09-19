# Install necessary libraries
# !pip install transformers datasets

# Import relevant modules
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load a pretrained model and tokenizer (e.g., BERT for sequence classification)
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load the IMDb dataset
dataset = load_dataset('imdb')

# Convert dataset to Pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Perform train-test split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Convert back to Hugging Face dataset
from datasets import Dataset

train_data = Dataset.from_pandas(train_data)
test_data = Dataset.from_pandas(test_data)

# Apply preprocessing
train_data = train_data.map(preprocess_function, batched=True)
test_data = test_data.map(preprocess_function, batched=True)

import os

import torch
from transformers import (AutoModelForSequenceClassification, Trainer,
                          TrainingArguments)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Disable parallelism warning and MLflow logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MLFLOW_TRACKING_URI"] = "disable"
os.environ["HF_MLFLOW_LOGGING"] = "false"

# Ensure CPU usage if no GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a smaller, faster model like DistilBERT
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

# Use a subset of the dataset to speed up training
train_data = train_data.select(range(1000))  # Select 1000 samples for training
test_data = test_data.select(range(200))     # Select 200 samples for evaluation

# Set up training arguments for faster training
training_args = TrainingArguments(
    output_dir=os.path.join(script_dir, 'results'),
    eval_strategy="steps",
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=8,   
    num_train_epochs=1,              
    weight_decay=0,                  
    logging_steps=500,               
    save_steps=1000,                 
    save_total_limit=1,              
    gradient_accumulation_steps=1,   
    fp16=False,                      
    report_to="none",                
)

# Define the Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Print evaluation results
print(f"Accuracy: {results}")
