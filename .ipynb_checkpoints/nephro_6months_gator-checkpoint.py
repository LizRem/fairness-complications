from datasets import load_dataset, Dataset, DatasetDict
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModel, AutoConfig, MegatronBertForSequenceClassification, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import EarlyStoppingCallback
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample, shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, multilabel_confusion_matrix, average_precision_score

# data
df = pd.read_csv("nephropathy_6_month.csv", index_col=0)

# rename
df = df.rename(columns={'ckd': 'labels', 
                       'aggregated_terms': 'text'})

df['text'] = df['text'].str.lower()
print(df['labels'].value_counts())

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


# downsample the train dataset
no_ckd = train_df[train_df['labels'] == 0] 
ckd = train_df[train_df['labels'] == 1]

# upsample minority class
downsample = resample(no_ckd, # what you want to downsample 
                    random_state=42, 
                    n_samples=len(ckd) # the length you want
                    )

# join back together
balanced_df = pd.concat([downsample, ckd])

print("Training dataset...", Counter(balanced_df['labels']))
print("Test dataset...", Counter(test_df['labels']))
print("Validation dataset...", Counter(val_df['labels']))

# group data together
ckd_df = DatasetDict({
    'train': Dataset.from_pandas(balanced_df),
    'test': Dataset.from_pandas(test_df),
    'valid': Dataset.from_pandas(val_df)})

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['labels']),
    y=train_df['labels']
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class weights:", class_weights)

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float(), 
                                                      pos_weight=self.class_weights[1], 
                                                      reduction='mean')
        else:
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float(), 
                                                      reduction='mean')
        
        return (loss, outputs) if return_outputs else loss
    
# define model and tokenizer
model_name = "UFNLP/gatortron-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_len=512
model = MegatronBertForSequenceClassification.from_pretrained(model_name,
                                                           num_labels=1,
                                                           problem_type="single_label_classification")

# tokenizer.truncation_side="left"

# tokenise dataset
def tokenize(batch):
  return tokenizer(batch["text"], 
                   truncation=True, 
                   padding=True,
                   max_length=512)

tokenized_dataset = ckd_df.map(tokenize, batched=True)

# set padding using dynamic padding
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# define Trainer
args = TrainingArguments(
    output_dir="/data/scratch/hhz049/a_nephro_6months_gator",
    per_device_train_batch_size=8, # 8 default
    per_device_eval_batch_size=8, # 8 default
    evaluation_strategy="steps",
    eval_steps=1000, # 4 times per epoch  
    save_strategy="steps",
    save_steps=2000,  
    logging_strategy="steps",
    logging_steps=1000,
    max_steps=12000, # downsampling train
    fp16=True,
    seed=42,
    learning_rate=2e-5, 
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    optim="adamw_torch",
    lr_scheduler_type="linear",
    gradient_accumulation_steps=1
)

# move model and data to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Modify the compute_metrics function to handle binary classification
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits > 0).astype(int).flatten()
    probs = torch.sigmoid(torch.tensor(logits)).numpy().flatten()

    accuracy_value = accuracy_score(labels, predictions)
    precision_value = precision_score(labels, predictions, average='binary', zero_division=0)
    recall_value = recall_score(labels, predictions, average='binary', zero_division=0)
    f1_value = f1_score(labels, predictions, average='binary', zero_division=0)
    auprc_value = average_precision_score(labels, probs)
    auroc_value = roc_auc_score(labels, probs)

    return {
        "accuracy": accuracy_value,
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1_value,
        "auprc": auprc_value,
        "auroc": auroc_value
    }

# Update the trainer instantiation
trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], # was 5 
    class_weights=class_weights  # Pass the calculated class weights
)

# Fine tune pre-trained model
trainer.train()

# Get training and evaluation loss from log history
loss_values = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
eval_loss_values = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]

# training and evaluation loss values
print("Training Loss Values:")
print(loss_values)

print("Evaluation Loss Values:")
print(eval_loss_values)

# apply to test set
eval_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
print("Evaluation results...", eval_results)
print(f"AUPRC: {eval_results['eval_auprc']:.4f}")
print(f"AUROC: {eval_results['eval_auroc']:.4f}")

# Get predictions for the test set
predictions, labels, _ = trainer.predict(tokenized_dataset["test"])
predictions = (predictions > 0).astype(int).flatten()

# Classification report
print("Classification Report:")
print(classification_report(labels, predictions, target_names=["No CKD", "CKD"], zero_division=0))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(labels, predictions))

# save model
trainer.save_model("/data/scratch/hhz049/a_nephro_6months_gator")
tokenizer.save_pretrained("/data/scratch/hhz049/a_nephro_6months_gator")
