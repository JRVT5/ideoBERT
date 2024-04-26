from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import pandas as pd

def filter_short_text(df):
    # Filter out rows where the text column has less than 10 words
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df_filtered = df[df['word_count'] >= 10]
    
    # Drop the word_count column
    df_filtered = df_filtered.drop(columns=['word_count'])
    
    return df_filtered

def get_dataloader(data_split: str, data_path: str = None, batch_size: int = 4):
    """
    Get a pytorch dataloader for a specific data split

    Args:
        data_split (str): the data split
        data_path (str, optional): a data path if the data is not stored at the default path.
            For students using ada, this should always be None. Defaults to None.
        batch_size (int, optional): the desired batch size. Defaults to 4.

    Returns:
        DataLoader: the pytorch dataloader object
    """
    assert data_split in ["train_partial", "dev_partial", "test"]
    if data_path is None:
        df = pd.read_csv(f"/storage/homes/jbroberts/FinalProject/data/{data_split}.csv", sep=",")
    else:
        df = pd.read_csv(data_path, sep=",")
    data = filter_short_text(df)
    data["label"] = data["label"].apply(lambda x: 1 if x == "conservative" else 0)
    dataset = Dataset.from_pandas(data)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", model_max_length=512)
    tokenized_train = dataset.map(lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length"), batched=True)
    # convert to pytorch format
    tokenized_train = tokenized_train.with_format("torch")
    return tokenized_train

train_dataloader = get_dataloader("train_partial", batch_size=16)
dev_dataloader = get_dataloader("dev_partial", batch_size=16)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

training_args = TrainingArguments(
    output_dir="/storage/homes/jbroberts/FinalProject/output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    logging_dir="/storage/homes/jbroberts/FinalProject/logs",
    logging_steps=100,
    eval_steps=200,
    save_strategy="epoch",
    save_steps=500,
    warmup_steps=500,
)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=dev_dataloader,
    compute_metrics=compute_metrics,
)
  
if __name__ == "__main__":
    trainer.train()
