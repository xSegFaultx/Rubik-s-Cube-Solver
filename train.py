import numpy as np
import torch
import pickle
from dataloader import CubeDataset
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from model import Model
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # determine if GPU is available

train_num = 90000
batch_size = 2048
lr = 0.0001
epochs = 800

with open("train_data_1.pkl", "rb") as train_file:
    x = pickle.load(train_file)
with open("label_data_1.pkl", "rb") as label_file:
    y = pickle.load(label_file)
x = np.array(x)
y = np.array(y)
train_x = x[:train_num]
valid_x = x[train_num:]
train_y = y[:train_num]
valid_y = y[train_num:]
train_dataset = CubeDataset(train_x, train_y)
valid_dataset = CubeDataset(valid_x, valid_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
train_loader_len = len(train_loader)

model = Model()
model = model.to(device)
model.train()

criterion = nn.MSELoss()
# optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(params=model.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999, verbose=True)
best_loss = 100


def evaluate(model, loader):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    counter = 0
    with torch.no_grad():
        for sample in loader:
            x = sample["x"]
            y = sample["y"]
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = torch.sqrt(criterion(y_hat, y))
            y_hat_choice = torch.max(y_hat, dim=1).indices
            y_choice = torch.max(y, dim=1).indices
            correct = torch.sum(y_hat_choice == y_choice)
            total_accuracy += correct.item() / x.shape[0]
            counter += 1
            total_loss += loss.item()
    average_loss = total_loss / counter
    average_accuracy = total_accuracy/counter
    return average_loss, average_accuracy

train_l = []
train_a = []
valid_l = []
valid_a = []
print("Training Start")
for epoch in range(epochs):
    running_loss = 0
    running_accuracy = 0
    for index, sample in enumerate(train_loader):
        x = sample["x"]
        y = sample["y"]
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        y_hat = model(x)
        loss = torch.sqrt(criterion(y_hat, y))
        y_hat_choice = torch.max(y_hat, dim=1).indices
        y_choice = torch.max(y, dim=1).indices
        correct = torch.sum(y_hat_choice == y_choice)
        running_accuracy += correct.item() / x.shape[0]
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    valid_loss, valid_accuracy = evaluate(model, valid_loader)
    print("Epoch {}/{}  Training Loss: {} Training Accuracy: {} Valid Loss: {} Valid Accuracy: {}".format(epoch + 1, epochs,
                                                                                       running_loss / train_loader_len,
                                                                                       running_accuracy/train_loader_len,
                                                                                       valid_loss, valid_accuracy))
    train_l.append(running_loss / train_loader_len)
    train_a.append(running_accuracy/train_loader_len)
    valid_l.append(valid_loss)
    valid_a.append(valid_accuracy)
    model.train()
    if epoch > 50:
        # scheduler.step()
        if valid_loss<best_loss:
            best_loss = valid_loss
            #torch.save(model.state_dict(), "./model_weights_1.dat")
            #print("Model Saved")
with open("loss.pkl", "wb") as loss_file:
    pickle.dump([train_l, train_a, valid_l, valid_a], loss_file, protocol=pickle.HIGHEST_PROTOCOL)
