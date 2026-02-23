import torch
import torch.nn as nn


class TabTransformerModel(nn.Module):
    def __init__(
        self,
        num_categories,
        num_numeric,
        d_model=64,
        n_heads=4,
        n_layers=3,
        emb_dim=32,
    ):
        super().__init__()

        self.category_emb = nn.Embedding(num_categories, emb_dim)

        self.input_proj = nn.Linear(num_numeric + emb_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=128,
            dropout=0.1,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x_num, x_cat):
        cat_emb = self.category_emb(x_cat)

        x = torch.cat([x_num, cat_emb], dim=1)

        x = self.input_proj(x).unsqueeze(1)

        x = self.encoder(x)

        x = x[:, 0, :]

        return self.regressor(x)
