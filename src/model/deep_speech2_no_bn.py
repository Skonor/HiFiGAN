from torch import nn

from src.base import BaseModel


class DeepSpeech2NoBN(BaseModel):
    def __init__(self, n_feats, n_class, hidden_dim=800, num_rnns=5, **batch):
        super().__init__(n_feats, n_class, **batch)


        self.convs  = nn.Sequential(
                      nn.Conv2d(in_channels=1, out_channels=32, kernel_size=[11, 41], stride=[2, 2]),
                      nn.BatchNorm2d(32),
                      nn.Hardtanh(0, 20),
                      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[11, 21], stride=[1, 2], padding=[5, 0]),
                      nn.BatchNorm2d(32),
                      nn.Hardtanh(0, 20)
                    )

        self.rnns = nn.ModuleList()
        input_size = 32 * ((((n_feats - 41) // 2 + 1) - 21) // 2 + 1)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_rnns, bidirectional=True, batch_first=True, dropout=0.0)

        self.head = nn.Sequential(
            nn.Linear(in_features=2 * hidden_dim, out_features=2 * hidden_dim),
            nn.Hardtanh(0, 20),
            nn.Linear(in_features=2 * hidden_dim, out_features=n_class)
        )

    def forward(self, spectrogram, **batch):

        x = spectrogram.transpose(1, 2).unsqueeze(1) # (b, 1, time, n_feats)
        x = self.convs(x) # (b, 32, time, features)
        x = x.transpose(1, 2).flatten(2) # (b, time, features)
        x = self.rnn(x)[0] # (b, time, 1600)
        logits = self.head(x) # (batch, time, freq)
        
        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        conv_length = (input_lengths - 10 - 1) // 2 + 1
        return conv_length