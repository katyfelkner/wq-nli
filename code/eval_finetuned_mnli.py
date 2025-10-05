from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import torch
import numpy as np
import evaluate
import os
import pandas as pd
import argparse
import logging
import sys

# Define softmax for probability conversion
softmax = torch.nn.Softmax(dim=1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on MNLI dataset")
    parser.add_argument("--model_path", type=str, 
                        default="../new_finetune/bert-base-cased-raw-finetuned-mnli", 
                        help="Path to the fine-tuned model")
    parser.add_argument("--dataset_split", type=str, default="validation_matched", 
                        choices=["validation_matched", "validation_mismatched", "test_matched", "test_mismatched"],
                        help="Dataset split to evaluate on (default: validation_matched)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                        help="Directory to save results (default: evaluation_results)")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for evaluation (default: 8)")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting evaluation with output directory: {args.output_dir}")
    logger.info(f"Checking if output directory exists: {os.path.exists(args.output_dir)}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Dataset split: {args.dataset_split}")
    
    # Load dataset
    logger.info("Loading MNLI dataset")
    dataset = load_dataset("nyu-mll/multi_nli")
    
    # Load model and tokenizer separately
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    # Load tokenizer directly from Hugging Face
    logger.info("Loading tokenizer from 'bert-base-cased' on Hugging Face")
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer successfully loaded")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {str(e)}")
        raise
    
    # Load model from fine-tuned path
    logger.info(f"Loading model from: {args.model_path}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        logger.info("Model successfully loaded")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    # Get max position embeddings from model config
    max_length = model.config.max_position_embeddings
    logger.info(f"Using max sequence length from model config: {max_length}")
    
    # Prepare evaluation dataset
    logger.info(f"Preparing dataset split: {args.dataset_split}")
    eval_dataset = dataset[args.dataset_split]
    logger.info(f"Dataset size: {len(eval_dataset)}")
    
    # Map the dataset to apply tokenization
    logger.info("Tokenizing dataset")
    try:
        def tokenize_function(examples):
            return tokenizer(
                examples["premise"], 
                examples["hypothesis"], 
                padding="max_length", 
                truncation=True, 
                max_length=max_length
            )
        
        tokenized_dataset = eval_dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")  # Rename to match trainer expectations
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        logger.info(f"Tokenized dataset created with {len(tokenized_dataset)} samples")
    except Exception as e:
        logger.error(f"Error tokenizing data: {str(e)}")
        raise
    
    # Set up metrics
    logger.info("Setting up evaluation metrics")
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # Set up training arguments
    logger.info(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )
    
    # Set up trainer
    logger.info("Setting up trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Run predictions
    logger.info("Running evaluation...")
    try:
        predictions = trainer.predict(tokenized_dataset)
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise
    
    # Process predictions
    logger.info("Processing predictions")
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    
    # Calculate accuracy
    accuracy = (pred_labels == predictions.label_ids).mean()
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    # Save predictions to file
    logger.info("Preparing results dataframe")
    # Convert logits to probabilities with softmax
    probs = softmax(torch.tensor(predictions.predictions)).numpy().tolist()
    
    result_df = pd.DataFrame({
        'premise': eval_dataset['premise'],
        'hypothesis': eval_dataset['hypothesis'],
        'true_label': eval_dataset['label'],
        'predicted_label': pred_labels,
        'predicted_probs': probs
    })
    
    model_name = os.path.basename(args.model_path)
    result_path = os.path.join(args.output_dir, f"{model_name}_{args.dataset_split}_results.csv")
    
    logger.info(f"About to save results to: {result_path}")
    logger.info(f"Directory exists: {os.path.exists(os.path.dirname(result_path))}")
    try:
        result_df.to_csv(result_path, index=False)
        logger.info(f"Successfully saved results to {result_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
    
    # Save metrics summary
    metrics_dict = {
        'model': args.model_path,
        'dataset': args.dataset_split,
        'accuracy': float(accuracy),  # Convert numpy float to Python float for JSON serialization
    }
    
    metrics_path = os.path.join(args.output_dir, f"{model_name}_{args.dataset_split}_metrics.json")
    try:
        with open(metrics_path, 'w') as f:
            import json
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"Successfully saved metrics to {metrics_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {str(e)}")
    
    logger.info("Evaluation process completed")
    return metrics_dict

if __name__ == "__main__":
    main()
