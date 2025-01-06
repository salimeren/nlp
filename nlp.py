import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    LlamaTokenizer
)
from datasets import Dataset
import torch
import evaluate
from typing import Dict, List
import os
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParliamentaryClassification:
    def __init__(
        self, 
        country_code: str,
        masked_model_name: str = "xlm-roberta-base",
        llama_path = Path("/content/drive/MyDrive/Llama-2-7b"),
        max_length: int = 512,
        output_dir: str = "./results"
    ):
        self.country_code = country_code
        self.masked_model_name = masked_model_name
        self.llama_path = Path(llama_path)
        self.max_length = max_length
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading masked language model: {masked_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(masked_model_name)
        
        logger.info(f"Using device: {self.device}")
        
        logger.info("Loading Llama tokenizer")
        self.setup_llama_tokenizer()
        
        # Calculate class weights
        self.class_weights = None
    
    def setup_llama_tokenizer(self):
        try:
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.llama_path)
        except Exception as e:
            logger.error(f"Error loading Llama tokenizer: {e}")
            raise
    
    def calculate_class_weights(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        weights = torch.tensor(
            [counts.sum() / (len(unique_labels) * count) for count in counts],
            dtype=torch.float32
        ).to(self.device)
        return weights
    
    def load_data(self, task: str = "orientation") -> tuple:
        train_path = Path("./orientation-fr-train.tsv")
        logger.info(f"Loading {task} data for country {self.country_code}")
        
        df = pd.read_csv(train_path, sep='\t')
        
        train_df, test_df = train_test_split(
            df,
            test_size=0.1,
            stratify=df['label'],
            random_state=42
        )
        
        return train_df, test_df
    
    def prepare_datasets(self, train_df: pd.DataFrame, task: str) -> tuple:
        logger.info("Preparing datasets")
        
        # Balance dataset through oversampling
        df_majority = train_df[train_df.label==1]
        df_minority = train_df[train_df.label==0]
        
        df_minority_upsampled = resample(
            df_minority, 
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )
        
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
        train_df = df_balanced.sample(frac=1, random_state=42)
        
        try:
            train_data, val_data = train_test_split(
                train_df,
                test_size=0.1,
                stratify=train_df['label'],
                random_state=42
            )
            
            # Calculate class weights
            self.class_weights = self.calculate_class_weights(train_data['label'])
            
            train_dataset = self._create_dataset(train_data)
            val_dataset = self._create_dataset(val_data)
            
            train_dataset = self._tokenize_dataset(train_dataset)
            val_dataset = self._tokenize_dataset(val_dataset)
            
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Error in prepare_datasets: {e}")
            raise
    
    def _create_dataset(self, df: pd.DataFrame) -> Dataset:
        try:
            dataset_dict = {
                'text': df['text'].values,
                'labels': df['label'].values
            }
            return Dataset.from_dict(dataset_dict)
        except Exception as e:
            logger.error(f"Error in _create_dataset: {e}")
            raise
    
    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            lambda x: self.tokenizer(
                x['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            ),
            batched=True
        )
    
    def train_masked_model(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        task: str,
        num_epochs: int = 5,
        batch_size: int = 16
    ):
        logger.info(f"Training masked model for task: {task}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.masked_model_name,
            num_labels=2
        ).to(self.device)
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / f"{task}_model"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=1e-5,
            weight_decay=0.2,
            warmup_ratio=0.1,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            push_to_hub=False,
            report_to="none",
            logging_dir=str(self.output_dir / "logs"),
            logging_first_step=True,
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            metrics = {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, average='weighted'),
                "precision": precision_score(labels, predictions, average='weighted'),
                "recall": recall_score(labels, predictions, average='weighted')
            }
            
            print(f"\nPredictions distribution: {np.unique(predictions, return_counts=True)}")
            print(f"Labels distribution: {np.unique(labels, return_counts=True)}")
            print(f"Metrics: {metrics}")
            
            return metrics
        
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss
        
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.class_weights = self.class_weights
        
        try:
            result = trainer.train()
            logger.info(f"Training completed - metrics: {result.metrics}")
            trainer.save_model(str(self.output_dir / f"best_{task}_model"))
            return trainer
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
   
    
    def evaluate_predictions(self, true_labels: np.ndarray, predictions: np.ndarray):
        metrics = {
            "accuracy": accuracy_score(true_labels, predictions),
            "f1": f1_score(true_labels, predictions, average='weighted'),
            "precision": precision_score(true_labels, predictions, average='weighted'),
            "recall": recall_score(true_labels, predictions, average='weighted')
        }
        return metrics

def main():
    classifier = ParliamentaryClassification(country_code='fr')
    
    # Task 1: Orientation Classification
    '''logger.info("Starting orientation classification")
    train_df, test_df = classifier.load_data(task="orientation")
    train_dataset, val_dataset = classifier.prepare_datasets(train_df, task="orientation")
    
    trainer = classifier.train_masked_model(
        train_dataset,
        val_dataset,
        task="orientation",
        num_epochs=5,
        batch_size=16
    )'''
    
    
    
    # Task 2: Power Classification
    logger.info("Starting power classification")
    train_df, test_df = classifier.load_data(task="power")
    train_dataset, val_dataset = classifier.prepare_datasets(train_df, task="power")
    
    trainer = classifier.train_masked_model(
        train_dataset,
        val_dataset,
        task="power",
        num_epochs=5,
        batch_size=16
    )
    
   

if __name__ == "__main__":
    main()