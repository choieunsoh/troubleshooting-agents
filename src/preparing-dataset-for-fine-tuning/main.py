# Install modules
# A '!' in a Jupyter Notebook runs the line in the system's shell, and not in the Python interpreter

import os
import random

# Import necessary libraries
import pandas as pd

# Load dataset 
# you can download this dataset from https://huggingface.co/datasets/stepp1/tweet_emotion_intensity/tree/main
#data = pd.read_csv('data/tweet_emotion_intensity.csv')
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, 'data', 'train.csv')
data = pd.read_csv(data_path)

# Preview the data
print(data.head())

import re  # Import the `re` module for working with regular expressions


# Function to clean the text
def clean_text(text):
    text = text.lower() # Convert all text to lowercase for uniformity
    text = re.sub(r'http\S+', '', text) # Remove URLs from the text
    text = re.sub(r'<.*?>', '', text) # Remove any HTML tags from the text
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation, keep only words and spaces
    return text # Return the cleaned text

# Assume `data` is a pandas DataFrame with a column named 'text'
# Apply the cleaning function to each row of the 'text' column
data['cleaned_text'] = data['tweet'].apply(clean_text)

# Print the first 5 rows of the cleaned text to verify the cleaning process
print(data['cleaned_text'].head())

# Check for missing values in the dataset
print(data.isnull().sum()) # Print the count of missing values for each column

# Option 1: Remove rows with missing data in the 'cleaned_text' column
data = data.dropna(subset=['cleaned_text']) # Drop rows where 'cleaned_text' is NaN (missing)

# Option 2: Fill missing values in 'cleaned_text' with a placeholder
data['cleaned_text'].fillna('unknown', inplace=True) # Replace NaN values in 'cleaned_text' with 'unknown'

from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the cleaned text
tokens = tokenizer(
    data['cleaned_text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt'
)

print(tokens['input_ids'][:5])  # Preview the first 5 tokenized examples

# Import necessary modules
import random  # Random module for generating random numbers and selections

import nltk

nltk.download('wordnet')

from nltk.corpus import wordnet  # NLTK's WordNet corpus for finding synonyms


# Define a function to find and replace a word with a synonym
def synonym_replacement(word):
# Get all synsets (sets of synonyms) for the given word from WordNet
    synonyms = wordnet.synsets(word)

# If the word has synonyms, randomly choose one synonym, otherwise return the original word
    if synonyms:
# Select a random synonym and get the first lemma (word form) of that synonym
        return random.choice(synonyms).lemmas()[0].name()

# If no synonyms are found, return the original word
    return word

# Define a function to augment text by replacing words with synonyms randomly
def augment_text(text):
# Split the input text into individual words
    words = text.split() # Split the input text into individual words

# Replace each word with a synonym with a probability of 20% (random.random() > 0.8)
    augmented_words = [
    synonym_replacement(word) if random.random() > 0.8 else word 
# If random condition met, replace
for word in words] # Iterate over each word in the original text

# Join the augmented words back into a single string and return it
    return ' '.join(augmented_words)

# Apply the text augmentation function to the 'cleaned_text' column in a DataFrame
# Create a new column 'augmented_text' containing the augmented version of 'cleaned_text'
data['augmented_text'] = data['cleaned_text'].apply(augment_text)

import torch  # Import PyTorch library
from torch.utils.data import (  # Import modules to create datasets and data loaders
    DataLoader, TensorDataset)

# Convert tokenized data into PyTorch tensors
input_ids = tokens['input_ids'] # Extract input IDs from the tokenized data
attention_masks = tokens['attention_mask'] # Extract attention masks from the tokenized data

# Define a mapping function
def map_sentiment(value):
    if value == "high":
        return 1
    elif value == "medium":
        return 0.5
    elif value == "low":
        return 0
    else:
        return None  # Handle unexpected values, if any

# Apply the function to each item in 'sentiment_intensity'
data['sentiment_intensity'] = data['sentiment_intensity'].apply(map_sentiment)

# Drop any rows where 'sentiment_intensity' is None
data = data.dropna(subset=['sentiment_intensity']).reset_index(drop=True)

# Convert the 'sentiment_intensity' column to a tensor
labels = torch.tensor(data['sentiment_intensity'].tolist())

from sklearn.model_selection import \
    train_test_split  # Import function to split dataset

# First split: 15% for test set, the rest for training/validation
train_val_inputs, test_inputs, train_val_masks, test_masks, train_val_labels, test_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.15, random_state=42
)

# Second split: 20% for validation set from remaining data
train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
    train_val_inputs, train_val_masks, train_val_labels, test_size=0.2, random_state=42
)

# Create TensorDataset objects for each set, including attention masks
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

# Create DataLoader objects
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

print("Training, validation, and test sets are prepared with attention masks!")
