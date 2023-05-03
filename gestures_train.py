import numpy as np
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from models.st_gcn_aaai18 import *
from utils.auxiliary import load_yaml


class NpyDataset(Dataset):
    """
    Class to read .npy format skeleton dataset
    """

    def __init__(self, data_folder, mode="train"):
        self.mode = mode
        self.num_channels = 4
        for item in os.listdir(data_folder):
            # Load the data
            if item.endswith(".npy") and mode in item:
                self.data = np.load(os.path.join(data_folder, item), mmap_mode="r")
            # Load the labels
            elif item.endswith(".pkl") and mode in item:
                with open(os.path.join(data_folder, item), "rb") as f:
                    self.labels = pickle.load(f)
                self.labels = self.labels[1]
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.labels[idx]


def train_model():
    params = load_yaml("cfg/train.yaml")
    dataset, layout, batch_size, learning_rate, momentum, nesterov, weight_decay, EPOCHS = (
        params["dataset"],
        params["layout"],
        params["batch_size"],
        params["learning_rate"],
        params["momentum"],
        params["nesterov"],
        params["weight_decay"],
        params["epochs"],
    )
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_data = NpyDataset(dataset)
    val_data = NpyDataset(dataset, mode="val")
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=1)

    # Model
    graph_cfg = {"layout": layout}
    model = ST_GCN_18(
        val_data.num_channels, val_data.num_classes, graph_cfg, edge_importance_weighting=True, data_bn=True
    )
    model.to(device)

    # Loss Function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay
    )

    # Function to train during 1 epoch
    def train_one_epoch(epoch_index, tb_writer):
        correct = 0
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            prediction = torch.argmax(outputs, dim=1)
            correct += (prediction == labels).sum().item()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100  # loss per batch
                print("Current training loss: " + str(last_loss))
                tb_x = epoch_index * len(train_dataloader) + i + 1
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0
        print("Training Accuracy = " + str(correct / (batch_size * len(train_dataloader))))
        return last_loss

    # Training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
    best_vloss = 1_000_000.0
    for epoch_number in range(EPOCHS):
        print("EPOCH {}:".format(epoch_number + 1))
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)
        model.train(False)
        running_vloss = 0.0
        correct_val = 0
        cont = 0
        for i, vdata in enumerate(val_dataloader):
            cont += 1
            vinputs, vlabels = vdata
            print("The shape of the input is {}".format(vinputs.shape))
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            prediction = torch.argmax(voutputs, dim=1)
            correct_val += (prediction == vlabels).sum().item()
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()
        print("Validaton Accuracy = " + str(correct_val / (len(val_dataloader))))
        avg_vloss = running_vloss / (i + 1)
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))
        writer.add_scalars(
            "Training vs. Validation Loss", {"Training": avg_loss, "Validation": avg_vloss}, epoch_number + 1
        )
        writer.flush()
        #  Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = "model_{}_{}".format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train_model()
