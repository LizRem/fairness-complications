from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModel, AutoConfig, MegatronBertForSequenceClassification, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers.modeling_outputs import TokenClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, multilabel_confusion_matrix, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# load model 
model_path = "/data/scratch/hhz049/a_nephro_6months_gator" # this is your fine tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# data
df = pd.read_csv("/data/scratch/hhz049/1-diabetes-complications/1-single_diseases/nephropathy_6_month.csv", index_col=0) # this is the data that matches your model time frame
demo = pd.read_csv("/data/scratch/hhz049/data/demographics_clean_new.csv", index_col=0) # these are my demographics

# merge 
df = df.merge(demo[['patid', 'sex', 'ethnic_group', 'imd_quintile']], on='patid', how='left')

# convert all columns to strings
for col in df.columns:
    if col != 'labels':
        df[col] = df[col].astype(str)
    
# split data 
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

# create Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# define the tokenizer
def tokenize(batch):
    return tokenizer(batch["text"], 
                     truncation=True, 
                     padding=True, 
                     max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize, batched=True)

# set the format
for dataset in [tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset]:
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# define fairness definitions
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def demographic_parity(y_true, y_pred):
    return np.mean(y_pred)

# bootstrap gap function
def bootstrap_gap(y_true_ref, y_pred_ref, y_true_group, y_pred_group, metric_func, n_iterations=1000, alpha=0.05):
    n_ref, n_group = len(y_true_ref), len(y_true_group)
    gaps = []
    
    for _ in range(n_iterations):
        idx_ref = np.random.randint(0, n_ref, n_ref)
        idx_group = np.random.randint(0, n_group, n_group)
        
        try:
            boot_metric_ref = metric_func(y_true_ref[idx_ref], y_pred_ref[idx_ref])
            boot_metric_group = metric_func(y_true_group[idx_group], y_pred_group[idx_group])
            gaps.append(boot_metric_group - boot_metric_ref)
        except ValueError:
            continue
    
    if not gaps:
        return None, (None, None)

    mean_gap = np.mean(gaps)
    ci_lower, ci_upper = np.percentile(gaps, [alpha/2 * 100, (1 - alpha/2) * 100])
    return mean_gap, (ci_lower, ci_upper)

# analyze gaps function
def analyze_gaps(y_true, y_pred, group_labels, reference_group, metrics):
    ref_mask = group_labels == reference_group
    y_true_ref, y_pred_ref = y_true[ref_mask], y_pred[ref_mask]
    
    results = {}
    for group in np.unique(group_labels):
        if group == reference_group:
            continue
        
        group_mask = group_labels == group
        y_true_group, y_pred_group = y_true[group_mask], y_pred[group_mask]
        
        group_results = {}
        for metric_name, metric_func in metrics.items():
            gap, (ci_lower, ci_upper) = bootstrap_gap(
                y_true_ref, y_pred_ref, 
                y_true_group, y_pred_group, 
                metric_func
            )
            if gap is not None:
                group_results[metric_name] = {
                    'gap': round(gap, 2),
                    'ci_lower': round(ci_lower, 2),
                    'ci_upper': round(ci_upper, 2),
                    'significant': (ci_lower > 0) or (ci_upper < 0)
                }
            else:
                group_results[metric_name] = {
                    'gap': None,
                    'ci_lower': None,
                    'ci_upper': None,
                    'significant': None
                }
        
        results[group] = group_results
    
    return results

# define the metrics you want to analyze
metrics = {
    'sensitivity': recall_score,
    'specificity': specificity_score,
    'demographic_parity': demographic_parity
}

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# predictions
def get_predictions(model, dataset):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(dataset), 8):  # Batch size of 8
            batch = dataset[i:i+8]
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).flatten()
            predictions = (logits > 0).int().flatten()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)

predictions, labels, probabilities = get_predictions(model, tokenized_test_dataset)

# analyse gaps
demographic_groups = ['sex', 'ethnic_group', 'imd_quintile']

for group in demographic_groups:
    if group in test_df.columns:
        print(f"\n--- Stratified Analysis by {group.capitalize()} ---")
        
        # Determine reference group
        if group == 'ethnic_group':
            reference_group = 'White'
        elif group == 'sex':
            reference_group = test_df[group].mode()[0]  # most common sex
        elif group == 'imd_quintile':
            reference_group = '1'  # least deprived
        else:
            reference_group = test_df[group].mode()[0]  # most common value
        
        group_labels = test_df[group].values
        
        gap_results = analyze_gaps(labels, predictions, group_labels, reference_group, metrics)
        
        print(f"Reference group: {reference_group}")
        for compared_group, group_results in gap_results.items():
            print(f"\nGaps for {group} = {compared_group} vs {reference_group}:")
            for metric, results in group_results.items():
                if results['gap'] is not None:
                    print(f"{metric.capitalize()}: Gap = {results['gap']} " 
                          f"(95% CI: {results['ci_lower']} to {results['ci_upper']})")
                    print(f"Statistically significant: {results['significant']}")
                else:
                    print(f"{metric.capitalize()}: Could not be calculated")