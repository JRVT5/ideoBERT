import argparse
from tqdm import tqdm
import torch
from torch import cuda, manual_seed
import torch.nn as nn
from model import PoliticalTextClassification
from util import get_dataloader
from util import model_accuracy


def train_model(model, train_dataloader, dev_dataloader, epochs, learning_rate):
    """
    Trains model and prints accuracy on dev data after training

    Arguments:
        model (HateSpeechClassificationModel): the model to train
        train_dataloader (DataLoader): a pytorch dataloader containing the training data
        dev_dataloader (DataLoader): a pytorch dataloader containing the development data
        epochs (int): the number of epochs to train for (full iterations through the dataset)
        learning_rate (float): the learning rate

    Returns:
        float: the accuracy on the development set
    """
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    for epoch in tqdm(range(epochs), desc="Epoch loop"):
        model.train()  # Set model to training mode
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label_int"].to(device)
            
            probs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(probs, label.float().unsqueeze(1))  # Assuming probs is output of sigmoid
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    accuracy = model_accuracy(model=model, dataloader=dev_dataloader, device=device)
    print(accuracy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=2, type=int, 
                        help="The number of epochs to train for")
    parser.add_argument("--learning_rate", default=1e-2, type=float, 
                        help="The learning rate")
    parser.add_argument("--batch_size", default=8, type=int, 
                        help="The batch size")
    args = parser.parse_args()

    print(args.batch_size)

    # initialize model and dataloaders
    device = "cuda" if cuda.is_available() else "cpu"

    # seed the model before initializing weights so that your code is deterministic
    manual_seed(457)

    model = PoliticalTextClassification().to(device)
    train_dataloader = get_dataloader("train_partial", batch_size=args.batch_size)
    dev_dataloader = get_dataloader("dev_partial", batch_size=args.batch_size)

    train_model(model, train_dataloader, dev_dataloader,
                args.epochs, args.learning_rate)


if __name__ == "__main__":
    main()