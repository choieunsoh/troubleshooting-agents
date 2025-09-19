# Load pre-trained BERT model
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Step 1: Freeze all layers except the last one (classification head)
for param in model.base_model.parameters():
    param.requires_grad = False

# If you'd like to fine-tune additional layers (e.g., the last 2 layers), you can unfreeze those layers as well
for param in model.base_model.encoder.layer[-2:].parameters():
    param.requires_grad = True

from transformers import Trainer, TrainingArguments

# Step 1: Set training arguments for fine-tuning the model
training_args = TrainingArguments(
    output_dir='./results',             # Directory where results will be stored
    num_train_epochs=3,                 # Number of epochs (full passes through the dataset)
    per_device_train_batch_size=16,     # Batch size per GPU/CPU during training
    eval_strategy="epoch",        # Evaluate the model at the end of each epoch
)

# Step 2: Fine-tune only the final classification head (since earlier layers were frozen)
trainer = Trainer(
    model=model,                        # Pre-trained BERT model with frozen layers
    args=training_args,                 # Training arguments
    train_dataset=train_data,           # Training data for fine-tuning
    eval_dataset=val_data,              # Validation data to evaluate performance during training
)

# Step 3: Train the model using PEFT (this performs PEFT because layers were frozen in Step 1)
trainer.train()

# Evaluate the model
results = trainer.evaluate(eval_dataset=test_data)
print(f"Test Accuracy: {results['eval_accuracy']}")

# Example of adjusting learning rate for PEFT optimization
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=5e-5,  # Experiment with different learning rates
    num_train_epochs=5,
    per_device_train_batch_size=16,
)
