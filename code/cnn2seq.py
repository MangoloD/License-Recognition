import os
import numpy as np
import torch
import torch.nn as nn
from sample_data import Sampling
from torch.utils.data.dataloader import DataLoader
from utils import *


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 5 * 14, 128),
        )

    def forward(self, x):
        out = self.cnn_layer(x)
        out = out.reshape(x.size(0), -1)
        out = self.fc(out)
        return out


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=128, num_layers=1, batch_first=True
        )
        self.out = nn.Linear(128, 65)

    def forward(self, x):
        # [N, 128]-->[N, 1, 128]-->[N, 7, 128]
        x = x.reshape(-1, 1, 128).expand(-1, 7, 128)
        lstm, (_, _) = self.lstm(x)
        # [N, 7, 128]-->[N*7, 128]
        y = lstm.reshape(-1, 128)
        # [N*4, 128]-->[N*7, 10]
        out = self.out(y)
        # [N*7, 10]-->[N, 7, 10]
        out = out.reshape(-1, 7, 65)
        return out


class MainNet(nn.Module):

    def __init__(self):
        super(MainNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder


if __name__ == '__main__':
    # x = torch.randn(2, 3, 140, 440)
    # net = MainNet()
    # y = net(x)
    # print(y.shape)
    # exit()

    batch = 10
    epochs = 100
    save_path = r'../params/cnn2seq.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MainNet().to(device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
        print("loaded params success...")
    else:
        print("NO Params")

    train_data = Sampling(r'G:/DL_Data/Plate/train_plate')
    train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)

    val_data = Sampling(r'G:/DL_Data/Plate/validate_plate')
    val_loader = DataLoader(val_data, batch_size=batch, shuffle=False)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            # print(x.shape, y.shape)
            batch_x, batch_y = x.to(device), y.float().to(device)
            output = net(batch_x)
            loss = loss_func(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(net.state_dict(), save_path)
        print("Save...")
        for i, (val_x, val_y) in enumerate(val_loader):
            # print(x.shape, y.shape)
            val_batch_x, val_batch_y = val_x.to(device), val_y.float().to(device)
            val_output = net(val_batch_x)
            val_loss = loss_func(val_output, val_batch_y)

            if i % 5 == 0:
                val_label_y = torch.argmax(val_y, 2).detach().cpu().numpy()
                val_out_y = torch.argmax(val_output, 2).detach().cpu().numpy()

                val_accuracy = np.sum(val_out_y == val_label_y, dtype=np.float32) / (batch * 7)

                print(f"epoch: {epoch + 1}\t i: {i}\t loss:{val_loss.item():.4f}\t acc:{val_accuracy * 100:.2f}%")
                print(f"label_y: {label_to_str(val_label_y[0])}")
                print(f"out_y: {label_to_str(val_out_y[0])}")
