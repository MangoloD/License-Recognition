import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(420, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=256,
                            num_layers=2,
                            batch_first=True)  # (S,N,V) -> (N,S,V)

    def forward(self, x):
        # (N,3,140,440) -> (N,420,440) -> (N,440,420) -> (N*440,420) -> (N*440,128) -> (N,440,128) -> (N,128) -> (N,256)
        x = x.reshape(-1, 420, 440).permute(0, 2, 1)
        x = x.reshape(-1, 420)
        fc1 = self.fc1(x)
        fc1 = fc1.reshape(-1, 440, 128)
        lstm, (h_n, h_c) = self.lstm(fc1, None)
        out = lstm[:, -1, :]

        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True)
        self.out = nn.Linear(128, 36)

    def forward(self, x):
        # (N,256) -> (N,7,256) -> (N,7,128) -> (N*7,128) -> (N*7,36) -> (N,7,36)
        x = x.reshape(-1, 1, 256)
        x = x.expand(-1, 7, 256)
        lstm, (h_n, h_c) = self.lstm(x, None)
        y1 = lstm.reshape(-1, 128)
        out = self.out(y1)
        output = out.reshape(-1, 7, 36)
        return output


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)

        return decoder
