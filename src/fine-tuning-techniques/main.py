import re


# Custom function to clean finance-related text
def clean_finance_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '[NUM]', text)  # Replace numbers with [NUM]
    text = re.sub(r'\$+', '[CURRENCY]', text)  # Replace currency symbols with a placeholder
    return text

from transformers import (BertForSequenceClassification, Trainer,
                          TrainingArguments)

# Load the pretrained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    logging_dir='./logs',
)

# Initialize the Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Train the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate()
print(f"Test Accuracy: {results['eval_accuracy']}")

# Use hyperparameter search to find the best settings
best_model = trainer.hyperparameter_search(
    direction="maximize",
    n_trials=10
)
