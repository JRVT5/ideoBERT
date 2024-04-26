import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import PoliticalTextClassification
import itertools
from typing import Any, List, Optional


def model_accuracy(model: PoliticalTextClassification, dataloader: DataLoader, device):
    """Compute the accuracy of a binary classification model

    Args:
        model (HateSpeechClassificationModel): a hate speech classification model
        dataloader (DataLoader): a pytorch data loader to test the model with
        device (string): cpu or cuda, the device that the model is on

    Returns:
        float: the accuracy of the model
    """
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in dataloader:
            pred = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            correct += (batch["label_int"] == (pred.to("cpu").squeeze() > 0.5).to(int)).sum().item()
            total += batch["label_int"].shape[0]
        acc = correct / total
        return acc

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
    data["label_int"] = data["label"].apply(lambda x: 1 if x == "conservative" else 0)
    dataset = Dataset.from_pandas(data)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = dataset.map(lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length"), batched=True)
    dataset = dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataset

def confusion_matrix(y_true: List[Any], y_pred: List[float], threshold: float, labels: List[Any]) \
    -> List[List[int]]:
    """
    Builds a confusion matrix given predictions and a threshold for binary classification.
    Uses the labels variable for the row/column order.

    Args:
        y_true (List[Any]): True labels.
        y_pred (List[float]): Predicted probabilities.
        threshold (float): Threshold for binary classification.
        labels (List[Any]): The column/row labels for the matrix.

    Returns:
        List[List[int]]: The confusion matrix.
    """
    # Convert predicted probabilities to binary labels based on the threshold
    y_pred_binary = [1 if pred >= threshold else 0 for pred in y_pred]

    # Check that all of the labels in y_true and y_pred are in the header list
    for label in y_true + y_pred_binary:
        assert label in labels, f"All labels from y_true and y_pred should be in labels, missing {label}"

    # Initialize and map labels
    num_classes = len(labels)
    matrix = [[0] * num_classes for _ in range(num_classes)]
    label_to_index = {label: i for i, label in enumerate(labels)}

    # Fill in the confusion matrix
    for true_label, pred_label in zip(y_true, y_pred_binary):
        true_index = label_to_index[true_label]
        pred_index = label_to_index[pred_label]
        # Add value
        matrix[true_index][pred_index] += 1

    return matrix

def _create_row(data: List[Any], cell_width: int, row_label: Optional[str] = None) -> str:
    strings = [str(item) for item in data]
    end = "|".join(string.center(cell_width) for string in strings) + "|"
    if row_label:
        return "|" + row_label.center(cell_width) + "|" + end
    else:
        return "|" + " " * cell_width + "|" + end


def _create_line(cell_width: int, num_cols: int, vbar_edge: bool = True, vbar_sep: bool = True):
    edge = "|" if vbar_edge else "-"
    sep = "|" if vbar_sep else "-"
    return edge + sep.join("-" * cell_width for _ in range(num_cols)) + edge


def print_confusion_matrix(matrix: List[List[int]], labels: List[str],
                           cell_width: int = 5):
    """
    Prints a given confusion matrix

    Args:
        matrix (List[List[int]]): the confusion matrix
        labels (List[Any]): the column/rows labels for the matrix
    """
    # choose a longer cell width if the int values are too large
    longest_num_length = max(len(str(num)) for num
                             in itertools.chain.from_iterable(matrix))
    cell_width = max(longest_num_length + 2, cell_width)
    num_cols = len(labels) + 1

    matrix_strings = []

    # build header
    matrix_strings.append(_create_line(cell_width, num_cols, vbar_edge=False, vbar_sep=False))
    matrix_strings.append("|" + "Actual Label".center((cell_width + 1) * num_cols - 1) + "|")
    matrix_strings.append(_create_line(cell_width, num_cols, vbar_sep=False))
    matrix_strings.append(_create_row(labels, cell_width))

    # build table data
    for i, row in enumerate(matrix):
        matrix_strings.append(_create_line(cell_width, num_cols))
        matrix_strings.append(_create_row(row, cell_width, row_label=labels[i]))
    matrix_strings.append(_create_line(cell_width, num_cols, vbar_edge=False, vbar_sep=False))

    # add in header for the row labels by shifting everything over
    predicted_label = ["Predicted", "Label"]
    labeled_matrix_strings = []
    label_position = len(matrix_strings) // 2
    label_width = max(len(x) for x in predicted_label) + 2
    j = 0
    for i, row in enumerate(matrix_strings):
        if i >= label_position and j < len(predicted_label) :
            # adding in the label
            label = predicted_label[j]
            j += 1
            labeled_matrix_strings.append(f"|{label.center(label_width)}{row}")
        elif i in (0, 4, len(matrix_strings) - 1):
            # special rows
            labeled_matrix_strings.append(f"{'-' * (label_width + 1)}{row}")
        else:
            # most rows
            labeled_matrix_strings.append(f"|{' ' * label_width}{row}")

    # print the final table
    print("\n".join(labeled_matrix_strings))