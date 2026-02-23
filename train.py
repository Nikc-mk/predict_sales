import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

from model import TabTransformerModel


def train_model(df):

    cat_encoder = LabelEncoder()
    df["cat_id"] = cat_encoder.fit_transform(df["category"])

    y = df["target"].values.astype("float32")

    X_num = df.drop(columns=["category", "target", "cat_id"]).values.astype("float32")
    X_cat = df["cat_id"].values.astype("int64")

    dataset = TensorDataset(
        torch.tensor(X_num),
        torch.tensor(X_cat),
        torch.tensor(y),
    )

    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = TabTransformerModel(
        num_categories=len(cat_encoder.classes_),
        num_numeric=X_num.shape[1],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.HuberLoss()

    for epoch in range(5):
        total = 0
        for x_num, x_cat, target in loader:
            pred = model(x_num, x_cat).squeeze()

            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"Epoch {epoch}: {total:.4f}")

    return model, cat_encoder
