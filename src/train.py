"""Training script for GNN models on node classification."""
import argparse
import torch
from tqdm import tqdm
import copy
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import yaml

from models.gnn_stack import GNNStack
from utils.optimizer import build_optimizer
from evaluate import test


def train(dataset, args):
    """Train GNN model on node classification task."""
    # Load dataset
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = GNNStack(
        input_dim=dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        output_dim=dataset.num_classes,
        args=args
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Setup optimizer
    scheduler, opt = build_optimizer(args, model.parameters())

    # Training loop
    losses = []
    test_accs = []
    best_acc = 0
    best_model = None
    for epoch in tqdm(range(args.epochs), desc="Training"):
        model.train()
        total_loss = 0

        for batch in loader:
            opt.zero_grad()

            # Forward pass
            pred = model(batch)

            # Compute loss on training nodes
            pred = pred[batch.train_mask]
            label = batch.y[batch.train_mask]
            loss = model.loss(pred, label)

            # Backward pass
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch.num_graphs

        # Average loss
        total_loss /= len(loader.dataset)
        losses.append(total_loss)

        # Evaluate periodically
        if epoch % args.eval_every == 0:
            test_acc = test(loader, model)
            test_accs.append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model)
        else:
          test_accs.append(test_accs[-1])


        # Update scheduler
        if scheduler is not None:
            scheduler.step()

    return best_model, best_acc, test_accs, losses, loader

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def main():
    parser = argparse.ArgumentParser(description='Train GNN on Cora')

    # Config file
    parser.add_argument('--config',type=str, required=True,
                        help='Path to a YAML configuration file')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Transfer each key–value from the YAML into args
    for key, val in cfg.items():
        setattr(args, key, val)

    # Make sure we have some necessary paramters
    required_params = ['model_type', 'dataset', 'hidden_dim', 'num_layers']
    for param in required_params:
        if not hasattr(args, param):
            raise ValueError(f"Missing required parameter: {param}")
    if not getattr(args, 'eval_every', False):
        setattr(args, 'eval_every', 10)


    # Load the Planetoid dataset (e.g. Cora) into memory
    # The root directory is ./data/<dataset> by default
    if args.dataset.lower() == 'cora':
        dataset = Planetoid(root=f'./data/{args.dataset}', name=args.dataset)
    else:
        raise NotImplementedError("Unknown dataset")


    print(f"Model: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of nodes: {dataset[0].x.shape[0]}")
    print(f"Number of features: {dataset.num_node_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Test set size: {dataset[0]['test_mask'].sum().item()}")
    print()

    # Train model
    best_model, best_acc, test_accs, losses, loader = train(dataset, args)

    print(f"\nMaximum test accuracy: {max(test_accs):.4f}")
    print(f"Minimum loss: {min(losses):.4f}")

    # Save best model predictions
    test(loader, best_model, save_model_preds=True, model_type=args.model_type)

    # Save model
    torch.save(best_model.state_dict(),
               f'./results/{args.model_type}_best.pth')



    # Plot results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title(f'Training Loss - {args.model_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(test_accs)
    plt.title(f'Test Accuracy - {args.model_type}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(f'./results/{args.model_type}_training.png')
    plt.show()

if __name__ == '__main__':
    main()