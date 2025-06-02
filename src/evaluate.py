"""Evaluation utilities for node classification."""
import torch
import pandas as pd


def test(loader, test_model, is_validation=False, save_model_preds=False, model_type=None):
    """Evaluate model on test set."""
    test_model.eval()

    correct = 0
    # Note that Cora is only graph
    for data in loader:
        with torch.no_grad():
            # Get predictions
            # max(dim=1) returns values, indices tuple; only need indices
            pred = test_model(data).max(dim=1)[1]
            label = data.y

        # Select appropriate mask
        mask = data.val_mask if is_validation else data.test_mask

        # node classification: only evaluate on nodes in test set
        # Filter predictions and labels
        pred = pred[mask]
        label = label[mask]

        # Save predictions if requested
        if save_model_preds and model_type:
            save_predictions(pred, label, model_type)

        # Calculate accuracy
        correct += pred.eq(label).sum().item()

    total = 0
    for data in loader.dataset:
        total += torch.sum(data.val_mask if is_validation else data.test_mask).item()

    return correct / total if total > 0 else 0


def save_predictions(pred, label, model_type):
    """Save model predictions to CSV."""
    print(f"Saving Model Predictions for Model Type: {model_type}")

    data = {}
    data['pred'] = pred.view(-1).cpu().detach().numpy()
    data['label'] = label.view(-1).cpu().detach().numpy()

    df = pd.DataFrame(data=data)
    # Save locally as csv
    df.to_csv(f'results/CORA-Node-{model_type}.csv', sep=',', index=False)