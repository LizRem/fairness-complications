import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import Counter
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, MegatronBertForSequenceClassification

def process_dataframe(df, name):
    """Process a single dataframe through the complete pipeline"""
    # Rename columns
    df = df.rename(columns={'ckd': 'labels', 'aggregated_terms': 'text'})
    df['text'] = df['text'].str.lower()
    print(f"\n=== Processing {name} ===")
    print("Label distribution:", df['labels'].value_counts())
    
    # Split data
    train_df, test_val_df = train_test_split(df, 
                                            test_size=0.2, 
                                            random_state=42,
                                            shuffle=True,
                                            stratify=df["labels"])
    
    val_df, test_df = train_test_split(test_val_df, 
                                      test_size=0.5, 
                                      random_state=42,
                                      shuffle=True,
                                      stratify=test_val_df["labels"])
    
    # Balance training data
    no_ckd = train_df[train_df['labels'] == 0]
    ckd = train_df[train_df['labels'] == 1]
    
    # Downsample majority class
    downsample = resample(no_ckd,
                         random_state=42,
                         n_samples=len(ckd))
    
    # Create balanced dataset
    balanced_df = pd.concat([downsample, ckd])
    
    print("\nDataset sizes:")
    print("Training dataset:", Counter(balanced_df['labels']))
    print("Test dataset:", Counter(test_df['labels']))
    print("Validation dataset:", Counter(val_df['labels']))
    
    # Create dataset dictionary
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(balanced_df),
        'test': Dataset.from_pandas(test_df),
        'valid': Dataset.from_pandas(val_df)
    })
    
    return dataset_dict

def analyze_token_lengths(dataset_dict, name, tokenizer):
    """Analyze token lengths for a dataset"""
    def count_tokens(batch):
        outputs = tokenizer(batch["text"],
                          add_special_tokens=True,
                          padding=False,
                          truncation=False)
        return {"token_count": [len(ids) for ids in outputs['input_ids']]}
    
    # tokenize dataset
    tokenized_dataset = dataset_dict.map(count_tokens, batched=True)
    
    # combine all splits
    combined_dataset = concatenate_datasets([
        tokenized_dataset['train'],
        tokenized_dataset['test'],
        tokenized_dataset['valid']
    ])
    
    # calculate statistics
    all_token_counts = np.array(combined_dataset['token_count'])
    max_length = 512
    num_truncated = np.sum(all_token_counts > max_length)
    percent_truncated = (num_truncated / len(all_token_counts)) * 100
    
    stats = {
        'mean': np.mean(all_token_counts),
        'median': np.median(all_token_counts),
        'q1': np.percentile(all_token_counts, 25),
        'q3': np.percentile(all_token_counts, 75)
    }
    
    print(f"\n=== Token Statistics for {name} ===")
    print(f"Total number of texts: {len(all_token_counts):,}")
    print(f"Median tokens: {stats['median']:.1f} [{stats['q1']:.1f}, {stats['q3']:.1f}]")
    print(f"Min token count: {np.min(all_token_counts)}")
    print(f"Max token count: {np.max(all_token_counts)}")
    
    print("\nTruncation Analysis:")
    print(f"Number of texts that will be truncated: {num_truncated:,}")
    print(f"Percentage of texts that will be truncated: {percent_truncated:.2f}%")

# load data
dataframes = {
    'Nephro 6 months': pd.read_csv("/data/scratch/hhz049/1-diabetes-complications/1-single_diseases/nephropathy_6_month.csv", index_col=0),
    'Nephro 1 year': pd.read_csv("/data/scratch/hhz049/1-diabetes-complications/1-single_diseases/nephropathy_1_year.csv", index_col=0),
    'Nephro 3 years': pd.read_csv("/data/scratch/hhz049/1-diabetes-complications/1-single_diseases/nephropathy_3_year.csv", index_col=0),
    'Nephro 6 years': pd.read_csv("/data/scratch/hhz049/1-diabetes-complications/1-single_diseases/nephropathy_5_year.csv", index_col=0)
}

# model and tokenizer
model_name = "UFNLP/gatortron-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MegatronBertForSequenceClassification.from_pretrained(model_name,
                                                            num_labels=1,
                                                            problem_type="single_label_classification")

# process each dataframe
processed_datasets = {}
for name, df in dataframes.items():
    dataset_dict = process_dataframe(df, name)
    processed_datasets[name] = dataset_dict
    analyze_token_lengths(dataset_dict, name, tokenizer)
