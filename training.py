from torch_geometric.loader import DataLoader
import torch, os
from torch.optim import AdamW
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from data_processing import read_json, get_dataset, get_cause_relations, TRANSFORMER
from graph_models import CauseExtractor
from utils import HOME_DIR


DEVICE = torch.device("cpu")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model() -> None:
    """
    Trains a cause extraction model using the provided dataset.
    """
    train_file = "data/Subtask_1_train_real.json"
    train = read_json(train_file)
    get_cause_relations(train)
    train_dataset = get_dataset(train)

    val_file = "data/Subtask_1_dev.json"
    val = read_json(val_file)
    get_cause_relations(val)
    val_dataset = get_dataset(val)



    num_epochs = 200
    batch_size = 32
    learning_rate = 0.0001
    weight_decay = 1e-5



    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = CauseExtractor().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()
    best_metric = 0

    for epoch in range(num_epochs):
        for data in loader:
            model.train()
            data = data.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(data)
            loss = loss_func(y_pred, data.y)
            loss.backward()
            optimizer.step()
        model.eval()
        y_golds = []
        y_preds = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                y_pred = model(data)
                y_pred = torch.argmax(F.softmax(y_pred, dim=1), dim=1)
                y_preds.extend(y_pred.cpu().numpy())
                y_golds.extend(data.y.cpu().numpy())
            metric = f1_score(y_golds, y_preds)
            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), f"{HOME_DIR}models/new_best_model_" + TRANSFORMER + ".pth")
            if epoch % 10 == 0:
                print(
                    f"""Epoch {epoch+1}/{num_epochs}, Train Loss: {round(loss.item(), 3)}, Validation F1: {round(metric, 3)}"""
                )
    print("Best Validation F1:", round(best_metric, 3))
    model.load_state_dict(torch.load(f"{HOME_DIR}models/new_best_model_" + TRANSFORMER + ".pth"))


if __name__ == "__main__":
    train_model()
