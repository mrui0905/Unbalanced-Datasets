import pandas as pd
import torch
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('creditcard.csv') # Downloaded from Kaggle
    features = df.drop(columns=['Class'])
    labels = df['Class']

    features_np = features.to_numpy()
    features_t = torch.tensor(features_np, dtype=torch.float32)
    labels_np = labels.to_numpy()
    labels_t = torch.tensor(labels_np, dtype=torch.float32)

    torch.manual_seed(0)

    features_np = features_t.numpy()
    labels_np = labels_t.numpy()

    train_features_np, test_features_np, train_labels_np, test_labels_np = train_test_split(
        features_np, labels_np, test_size=0.1, stratify=labels_np, random_state=0
    )

    train_features = torch.tensor(train_features_np, dtype=features_t.dtype)
    test_features = torch.tensor(test_features_np, dtype=features_t.dtype)
    train_labels = torch.tensor(train_labels_np, dtype=labels_t.dtype)
    test_labels = torch.tensor(test_labels_np, dtype=labels_t.dtype)

    torch.save(
        {
            "train_features": train_features,
            "train_labels": train_labels,
            "test_features": test_features,
            "test_labels": test_labels
        },
        "creditcard.pt"
    )