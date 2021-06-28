import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(420, 128),  # 420数据长度
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True)

    def forward(self, x):
        # [N,3,140,440] --> [N,420,440] --> [N,440,420]
        x = x.reshape(-1, 420, 440)
        # [N,440,420] --> [N*440,420]
        x = x.reshape(-1, 420)
        # [N*440,420].[420,128]=[N*440,128]
        fc1 = self.fc1(x)
        # [N*440,128] --> [N,440,128]
        fc1 = fc1.reshape(-1, 440, 128)
        lstm, *_ = self.lstm(fc1)
        # [N,440,128] --> [N,128]
        out = lstm[:, -1, :]
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True)
        self.out = nn.Linear(128, 65)
        # self.out_province = nn.Linear(128, 31)
        # self.out_upper = nn.Linear(128, 24)
        # self.out_digits = nn.Linear(128, 10)

    def forward(self, x):
        # [N,128] --> [N,1,128]
        x = x.reshape(-1, 1, 128)
        # [N,1,128] --> [N,7,128]
        x = x.expand(-1, 7, 128)
        lstm, *_ = self.lstm(x)
        # [N,7,128] --> [N*7,128]
        y1 = lstm.reshape(-1, 128)
        # [N*7,128].[128,65]=[N*7,65]
        out = self.out(y1)
        # out_province = self.out_province(y1)
        # out_upper = self.out_upper(y1)
        # out_digits = self.out_digits(y1)

        # [N*7,65] --> [N,7,65]
        output = out.reshape(-1, 7, 65)
        return output
        # return out_province, out_upper, out_digits


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
    a = torch.randn((2, 3, 140, 440))
    b = MainNet()
    print(b(a).shape)
