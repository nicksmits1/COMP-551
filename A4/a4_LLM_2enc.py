# -*- coding: utf-8 -*-
"""LLM Fine-Tuning with Classifier Head and Two Encoder Layers Trained"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np

"""Load Dataset"""
dataset = load_dataset('go_emotions')

"""Filter for singly labeled data"""
def has_single_label(example):
    return len(example['labels']) == 1

# Filter each split separately
train_dataset = dataset['train'].filter(has_single_label)
validation_dataset = dataset['validation'].filter(has_single_label)
test_dataset = dataset['test'].filter(has_single_label)

"""Map numerical labels to names"""
# Load label mapping
label_mapping = dataset['train'].features['labels'].feature.names

# Function to map label IDs to names
def map_label(example):
    label_id = example['labels'][0]
    example['label_name'] = label_mapping[label_id]
    example['label'] = label_id  # Add a single label field
    return example

# Apply the mapping to each split
train_dataset = train_dataset.map(map_label)
validation_dataset = validation_dataset.map(map_label)
test_dataset = test_dataset.map(map_label)

"""Tokenize Text for LLM"""
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
validation_dataset = validation_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

"""Prepare for Training"""
# PyTorch Format
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Number of emotion labels
num_labels = len(label_mapping)

"""Define Metrics and Classification Report"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_micro': f1_score(labels, predictions, average='micro'),
        'f1_macro': f1_score(labels, predictions, average='macro'),
    }


def print_classification_report(eval_pred):
    predictions = eval_pred.predictions  
    labels = eval_pred.label_ids  
    predictions = np.argmax(predictions, axis=-1)  
    print(classification_report(labels, predictions, target_names=label_mapping))

"""Train Model with Classifier Head and Two Encoder Layers Trained"""
print("Training model with classifier head and two encoder layers...")

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Freeze all layers except the last two encoder layers and the classification head
for name, param in model.base_model.named_parameters():
    if not (name.startswith("encoder.layer.10") or name.startswith("encoder.layer.11")):
        param.requires_grad = False

# Training arguments
training_args = TrainingArguments(
    output_dir='./results_two_layers_trained',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=16,  # Best batch size from tuning
    per_device_eval_batch_size=16,  
    logging_dir='./logs_two_layers_trained',
    logging_steps=100,
    learning_rate=5e-5,  # Best learning rate from tuning
    load_best_model_at_end=True,
    metric_for_best_model='f1_micro',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on the test set
test_results = trainer.evaluate(test_dataset)
print("Test Results for Two Encoder Layers Trained Model:", test_results)

# Print classification report for the test set
test_eval_pred = trainer.predict(test_dataset)
print("Classification Report for Two Encoder Layers Trained Model:")
print_classification_report(test_eval_pred)




