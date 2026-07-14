import torch
import argparse
import os
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import Dataset
from huggingface_hub import hf_hub_download, login
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from evaluation.masker import ProteinMasker
from speedrunning_plms.evaluation import (
    download_dataset_split,
    load_benchmark_manifest,
    load_benchmark_model,
    load_benchmark_tokenizer,
)
from speedrunning_plms.training.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument(
        '--manifest',
        type=str,
        default=str(Path(__file__).with_name('benchmark_manifest.json')),
        help='Immutable benchmark asset manifest',
    )
    return parser.parse_args()


class ProteinDataset(TorchDataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class ProteinCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.masker = ProteinMasker(tokenizer, mask_rate=0.15)

    def __call__(self, batch):
        tokenized_batch = self.tokenizer(
            batch,
            padding='longest',
            max_length=1022,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        tokenized_batch['input_ids'], tokenized_batch['labels'] = self.masker(tokenized_batch['input_ids'], tokenized_batch['attention_mask'])
        return tokenized_batch


def calculate_metrics(preds, labels):
    """Calculate metrics only where labels != -100"""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    # Create mask for valid positions (labels != -100)
    valid_mask = labels != -100
    
    if not valid_mask.any():
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mcc': 0.0,
            'num_tokens': 0
        }
    
    # Extract valid predictions and labels
    valid_preds = preds[valid_mask]
    valid_labels = labels[valid_mask]
    
    # Calculate metrics
    accuracy = accuracy_score(valid_labels, valid_preds)
    precision = precision_score(valid_labels, valid_preds, average='weighted', zero_division=0)
    recall = recall_score(valid_labels, valid_preds, average='weighted', zero_division=0)
    f1 = f1_score(valid_labels, valid_preds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(valid_labels, valid_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'num_tokens': len(valid_labels)
    }


def main():
    args = parse_args()
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Login once if token is provided
    if args.hf_token is not None:
        login(args.hf_token)
    
    # Initialize components that don't need to be recreated for each model or dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest = load_benchmark_manifest(args.manifest)
    tokenizer_asset = manifest['tokenizer']

    all_results = []

    for dataset_asset in manifest['datasets']:
        dataset_name = dataset_asset['name']
        for split_type in ['valid', 'test']:
            local_file = download_dataset_split(
                dataset_asset,
                split_type,
                downloader=hf_hub_download,
            )
            data = Dataset.from_parquet(local_file)
            print(f"Loaded {dataset_name} {split_type}: {len(data)} sequences")
            sequences = data['sequence']
            sequences = sorted(sequences, key=len, reverse=True)
            #sequences = sequences[-100:]  # Uncomment for debugging with smaller subset
            print(f"Shortest sequence: {len(sequences[-1])} tokens")

            for model_asset in manifest['models']:
                model_name = model_asset['repo_id']
                nickname = model_asset['nickname']
                print(f"\nEvaluating {nickname} on {dataset_name} {split_type}")
                set_seed(42)

                model = load_benchmark_model(
                    model_asset,
                    auto_model_cls=AutoModelForMaskedLM,
                ).to(device).eval()
                tokenizer = load_benchmark_tokenizer(
                    tokenizer_asset,
                    auto_tokenizer_cls=AutoTokenizer,
                )

                collator = ProteinCollator(tokenizer)
                dataset = ProteinDataset(sequences)
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    collate_fn=collator,
                    num_workers=args.num_workers,
                )
                
                # Initialize accumulators
                total_loss = 0.0
                total_tokens = 0
                all_preds = []
                all_labels = []
                num_batches = 0
                
                for batch in tqdm(dataloader, total=len(dataloader), desc=f'{nickname} {dataset_name} {split_type}'):
                    # Move batch to device
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    
                    with torch.no_grad():
                        outputs = model(**batch)
                        labels = batch['labels'].cpu()
                        loss = outputs.loss.item()
                        logits = outputs.logits.cpu()
                        preds = logits.argmax(dim=-1)
                        
                        # Accumulate loss
                        total_loss += loss
                        num_batches += 1
                        
                        # Flatten predictions and labels for metric calculation
                        preds_flat = preds.flatten()
                        labels_flat = labels.flatten()
                        
                        # Only keep predictions and labels where labels != -100
                        valid_mask = labels_flat != -100
                        if valid_mask.any():
                            all_preds.append(preds_flat[valid_mask])
                            all_labels.append(labels_flat[valid_mask])
                            total_tokens += valid_mask.sum().item()
                
                # Calculate overall metrics
                if all_preds:
                    all_preds = torch.cat(all_preds)
                    all_labels = torch.cat(all_labels)
                    metrics = calculate_metrics(all_preds.numpy(), all_labels.numpy())
                else:
                    metrics = {
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'mcc': 0.0,
                        'num_tokens': 0
                    }
                
                # Calculate perplexity
                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else 0.0
                
                # Store results
                result = {
                    'model': nickname,
                    'model_path': model_name,
                    'model_revision': model_asset['revision'],
                    'dataset': dataset_name,
                    'dataset_revision': dataset_asset['revision'],
                    'tokenizer_revision': tokenizer_asset['revision'],
                    'split': split_type,
                    'loss': round(avg_loss, 3),
                    'perplexity': round(perplexity, 3),
                    'accuracy': round(metrics['accuracy'], 3),
                    'precision': round(metrics['precision'], 3),
                    'recall': round(metrics['recall'], 3),
                    'f1': round(metrics['f1'], 3),
                    'mcc': round(metrics['mcc'], 3),
                    'num_sequences': len(sequences),
                    'num_tokens': total_tokens,
                    'num_batches': num_batches
                }
                
                all_results.append(result)
                print(f"Results for {nickname} on {dataset_name} {split_type}:")
                print(f"  Loss: {result['loss']:.4f}")
                print(f"  Perplexity: {result['perplexity']:.4f}")
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(f"  F1: {result['f1']:.4f}")
                print(f"  MCC: {result['mcc']:.4f}")
                print(f"  Tokens: {result['num_tokens']:,}")
                
                # Clean up GPU memory
                model.cpu()
                del model, tokenizer, collator
                torch.cuda.empty_cache()

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(args.results_dir, 'benchmark_results_esm.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(results_df.groupby(['model', 'dataset', 'split'])[['loss', 'perplexity', 'accuracy', 'f1', 'mcc']].mean().round(4))


if __name__ == '__main__':
    main()
