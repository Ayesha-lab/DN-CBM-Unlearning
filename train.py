from Models.mlp import MLP
import mlflow
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm


def train_model(model_state, train_loader, lr, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'training using: {device}', flush=True)

    mlflow.log_params({"lr": lr, "epochs": epochs})

    model = MLP(input_dim=784, output_dim=2)  # Create fresh model
    model.load_state_dict(model_state)
    model = model.to(device)  # Move model to GPU
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=0.0005)

    # ADDED: Track max gradient norm during training
    max_grad_norm = 0.0

    # mlflow context and experiment
    # itertools loop
    train_iter = tqdm(range(epochs),
                      desc="Training",
                      unit="epoch")

    for epoch in train_iter:
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Reshape and move data to GPU
            X_batch = X_batch.view(X_batch.size(0), -1).to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # ADDED: Compute gradient norm before step
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            max_grad_norm = max(max_grad_norm, float(grad_norm))
            mlflow.log_metric("grad_norm", grad_norm, step=epoch * len(train_loader) + batch_idx)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        train_iter.set_postfix({
            "avg_loss": f"{avg_loss:.4f}",
            "accuracy": f"{accuracy:.4f}"
        })

        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("train_accuracy", accuracy, step=epoch)

    # ADDED: Compute C0 from final model weights and C1 from max gradient norm
    num_params = sum(p.numel() for p in model.parameters()) ** 0.5
    model_norm = torch.norm(torch.cat([p.flatten() for p in model.parameters()]))/num_params
    c_0 = float(model_norm.detach() * 1.5)
    c_1 = max(max_grad_norm * 1.5, 1.0)

    print(f"Computed clipping thresholds: C0={c_0:.4f}, C1={c_1:.4f}", flush=True)

    return model, c_0, c_1