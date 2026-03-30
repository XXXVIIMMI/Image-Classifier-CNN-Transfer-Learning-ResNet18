from tqdm import tqdm


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):

    model.train()
    total_loss    = 0
    total_correct = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        logits = model(X)
        loss   = loss_fn(logits, y)

        # 2. Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. Track metrics
        total_loss    += loss.item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy


def eval_one_epoch(model, dataloader, loss_fn, device):

    model.eval()
    total_loss    = 0
    total_correct = 0

    with torch.inference_mode():   # faster than no_grad
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss   = loss_fn(logits, y)

            total_loss    += loss.item()
            total_correct += (logits.argmax(dim=1) == y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy


def train(model, train_loader, val_loader, loss_fn, optimizer, epochs, device):


    results = {
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    [],
    }

    for epoch in tqdm(range(epochs), desc="Training"):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, loss_fn, device
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}   | val_acc: {val_acc:.4f}"
        )

    return results
