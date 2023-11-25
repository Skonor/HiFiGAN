from torch import nn

from src.base import BaseModel

class Ds2Gru(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()

        self.grus = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.grus.append(nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(2 * hidden_size))
        for _ in range(num_layers - 1):
            self.grus.append(nn.GRU(input_size=2*hidden_size, hidden_size=hidden_size, batch_first=True, bidirectional=True, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(2 * hidden_size))

    def forward(self, x):
        for gru, bn in zip(self.grus, self.bns):
            x = gru(x)[0] # (b, time, features)
            x = bn(x.transpose(1, 2)).transpose(1, 2)  
        return x


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, hidden_dim=800, num_rnns=5, **batch):
        super().__init__(n_feats, n_class, **batch)


        self.convs  = nn.Sequential(
                      nn.Conv2d(in_channels=1, out_channels=32, kernel_size=[11, 41], stride=[2, 2], padding=[5, 20]),
                      nn.BatchNorm2d(32),
                      nn.Hardtanh(0, 20),
                      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[11, 21], stride=[1, 2], padding=[5, 10]),
                      nn.BatchNorm2d(32),
                      nn.Hardtanh(0, 20)
                    )

        self.rnns = nn.ModuleList()
        
        input_size_conv1 = (n_feats + 2 * self.convs[0].padding[1] - self.convs[0].kernel_size[1]) // 2 + 1
        input_size = 32 * ((input_size_conv1 + 2 * self.convs[3].padding[1] - self.convs[3].kernel_size[1]) // 2 + 1)
        self.rnn = Ds2Gru(input_size=input_size, hidden_size=hidden_dim, num_layers=num_rnns)

        self.lin = nn.Linear(in_features=2 * hidden_dim, out_features=2 * hidden_dim)
        self.bn_lin = nn.BatchNorm1d(2 * hidden_dim)
        self.head = nn.Sequential(
            nn.Hardtanh(0, 20),
            nn.Linear(in_features=2 * hidden_dim, out_features=n_class)
        )

    def forward(self, spectrogram, **batch):

        x = spectrogram.transpose(1, 2).unsqueeze(1) # (b, 1, time, n_feats)
        x = self.convs(x) # (b, 32, time, features)
        x = x.transpose(1, 2).flatten(2) # (b, time, features)
        x = self.rnn(x) # (b, time, 1600)
        x = self.lin(x)
        x = self.bn_lin(x.transpose(1, 2)).transpose(1, 2)
        logits = self.head(x)# (batch, time, freq)

        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        conv1_length = (input_lengths + 2 * self.convs[0].padding[0] - self.convs[0].kernel_size[0]) // 2 + 1
        conv2_length = (conv1_length + 2 * self.convs[3].padding[0] - self.convs[3].kernel_size[0]) + 1
        return conv2_length