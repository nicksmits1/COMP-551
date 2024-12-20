# -*- coding: utf-8 -*-
"""LLM Fine-Tuning and Pretrained Model Testing with Classification Reports"""

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
# Set format for PyTorch
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
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    print(classification_report(labels, predictions, target_names=label_mapping))

"""Test Pretrained Model Without Modification"""
print("Testing pretrained model without fine-tuning...")
pretrained_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Trainer for the pretrained model
pretrained_trainer = Trainer(
    model=pretrained_model,
    compute_metrics=compute_metrics,
)

# Evaluate the pretrained model on the test set
pretrained_results = pretrained_trainer.evaluate(test_dataset)
print("Pretrained Model Results:", pretrained_results)

# Print classification report
pretrained_eval_pred = pretrained_trainer.predict(test_dataset)
print("Classification Report for Pretrained Model:")
print_classification_report(pretrained_eval_pred)

"""Hyperparameter Tuning"""
learning_rates = [5e-5, 3e-5, 2e-5]
batch_sizes = [16, 32]
best_f1_micro = 0
best_config = None
results = []

for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"Training with learning rate {lr} and batch size {batch_size}")

        # Load pre-trained model
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

        # Model 1: Freeze all layers except classification head
        for param in model.base_model.parameters():
            param.requires_grad = False

        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            num_train_epochs=3,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_dir='./logs',
            logging_steps=100,
            learning_rate=lr,
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

        # Evaluate on the validation set
        eval_results = trainer.evaluate(validation_dataset)
        results.append({'learning_rate': lr, 'batch_size': batch_size, **eval_results})

        # Track the best configuration
        if eval_results['eval_f1_micro'] > best_f1_micro:
            best_f1_micro = eval_results['eval_f1_micro']
            best_config = {'learning_rate': lr, 'batch_size': batch_size}

# Print best configuration
print("Best Configuration:", best_config)

"""Train and Evaluate with Best Configuration"""
# Train both Model 1 and Model 2 using the best hyperparameters
for model_type in ["classification_head_only", "classification_and_last_layer"]:
    print(f"Training {model_type} model with best configuration")

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    if model_type == "classification_and_last_layer":
        # Freeze all layers except the last encoder layer and the classification head
        for name, param in model.base_model.named_parameters():
            if not name.startswith("encoder.layer.11"):  # Adjust for BERT's last layer
                param.requires_grad = False
    else:
        # Freeze all layers except the classification head
        for param in model.base_model.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=f'./results_{model_type}',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=3,
        per_device_train_batch_size=best_config['batch_size'],
        per_device_eval_batch_size=best_config['batch_size'],
        logging_dir=f'./logs_{model_type}',
        logging_steps=100,
        learning_rate=best_config['learning_rate'],
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
    print(f"Test Results for {model_type}:", test_results)

    # Print classification report
    test_eval_pred = trainer.predict(test_dataset)
    print(f"Classification Report for {model_type} Model:")
    print_classification_report(test_eval_pred)

    # Save the model
    trainer.save_model(f'./fine_tuned_model_{model_type}')

