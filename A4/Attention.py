import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the dataset
dataset = load_dataset('go_emotions')

# Filter for single-label examples
def has_single_label(example):
    return len(example['labels']) == 1

test_dataset = dataset['test'].filter(has_single_label)

# Map labels
label_mapping = dataset['train'].features['labels'].feature.names

def map_label(example):
    label_id = example['labels'][0]
    example['label_name'] = label_mapping[label_id]
    example['label'] = label_id
    return example

test_dataset = test_dataset.map(map_label)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

test_dataset = test_dataset.map(tokenize_function, batched=True)

#  PyTorch format
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load the saved model( last_model IS THIS MODEL IS THE MODEL WITH 2 TRAINED ENCODER LAYERS FROM a4_LLM_2enc.py)
model = AutoModelForSequenceClassification.from_pretrained('./last_model', output_attentions=True)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Evaluate the model to get predictions
def get_predictions(test_dataset):
    model.eval()
    predictions, labels = [], []

    for example in test_dataset:
        input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        predictions.append(pred)
        labels.append(example['label'])

    return np.array(predictions), np.array(labels)

predictions, labels = get_predictions(test_dataset)

# Find all correct and incorrect indices
correct_indices = np.where(predictions == labels)[0]
incorrect_indices = np.where(predictions != labels)[0]

# Randomly select one correct and one incorrect example
if len(correct_indices) > 0:
    correct_index = int(random.choice(correct_indices))  # Convert to Python int
else:
    correct_index = 0  # Default fallback if no correct predictions

if len(incorrect_indices) > 0:
    incorrect_index = int(random.choice(incorrect_indices))  # Convert to Python int
else:
    incorrect_index = 0  # Default fallback if no incorrect predictions

# Create directories to save the attention heatmaps
os.makedirs("correct_example", exist_ok=True)
os.makedirs("incorrect_example", exist_ok=True)

# Function to plot attention heatmap for all layers
def plot_attention_for_all_layers(example_index, directory, title_prefix, max_tokens=20):
    example_index = int(example_index)  # Python int
    input_ids = torch.tensor(test_dataset[example_index]['input_ids']).unsqueeze(0).to(model.device)
    attention_mask = torch.tensor(test_dataset[example_index]['attention_mask']).unsqueeze(0).to(model.device)

    # Get attention outputs
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    attentions = outputs.attentions

    # Convert token IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Exclude [PAD] tokens
    tokens = [token for token, mask in zip(tokens, input_ids[0]) if token != "[PAD]"]

    # Retrieve true and predicted labels
    true_label = label_mapping[test_dataset[example_index]['label']]
    predicted_label = label_mapping[torch.argmax(outputs.logits, dim=1).item()]

    # Plot attention for each layer
    for layer_index, attention_matrix in enumerate(attentions):
        # Extract attention for the first head
        attention_matrix = attention_matrix[0, 0].detach().cpu().numpy()

        # Adjust tokens and attention matrix size
        attention_matrix = attention_matrix[:len(tokens), :len(tokens)]

        # Normalize attention for better visualization
        attention_matrix /= attention_matrix.sum(axis=-1, keepdims=True)

        #Title for Plot
        title = f"{title_prefix}\nLayer {layer_index + 1} | True Label: {true_label} | Predicted Label: {predicted_label}"

        # Plot the attention heatmap
        plt.figure(figsize=(15, 12))
        sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
        plt.title(title)
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(directory, f"{title_prefix.replace(' ', '_')}_Layer_{layer_index + 1}.png")
        plt.savefig(save_path)
        plt.close()

# Plot attention for a randomly selected correct example
plot_attention_for_all_layers(correct_index, "correct_example", title_prefix="Attention for Correct Example")

# Plot attention for a randomly selected incorrect example
plot_attention_for_all_layers(incorrect_index, "incorrect_example", title_prefix="Attention for Incorrect Example")

