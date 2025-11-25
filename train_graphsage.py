"""
GraphSAGE training on ogbn-products using PyTorch Geometric.
Structured for easy DeepSpeed integration.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
from tqdm import tqdm

# PyTorch Geometric imports
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# Data loading
import pandas as pd
import urllib.request
import zipfile


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class GraphSAGE(nn.Module):
    """GraphSAGE model - DeepSpeed ready (no device placement in __init__)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


# ============================================================================
# DATA LOADING
# ============================================================================

def download_ogbn_products(data_dir):
    """Download ogbn-products dataset"""
    os.makedirs(data_dir, exist_ok=True)
    extract_path = os.path.join(data_dir, 'products')

    if os.path.exists(extract_path):
        print(f"Dataset exists at {extract_path}")
        return extract_path

    zip_path = os.path.join(data_dir, 'products.zip')
    url = 'https://snap.stanford.edu/ogb/data/nodeproppred/products.zip'

    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    return extract_path


def load_data(data_dir):
    """Load ogbn-products dataset"""
    print("\n" + "="*80)
    print("LOADING OGBN-PRODUCTS")
    print("="*80)

    extract_path = download_ogbn_products(data_dir)

    # Load features
    print("Loading node features...")
    x = torch.from_numpy(pd.read_csv(
        os.path.join(extract_path, 'raw', 'node-feat.csv.gz'),
        compression='gzip', header=None).values).float()

    # Load edges
    print("Loading edges...")
    edge_index = torch.from_numpy(pd.read_csv(
        os.path.join(extract_path, 'raw', 'edge.csv.gz'),
        compression='gzip', header=None).values.T).long()

    # Load labels
    print("Loading labels...")
    y = torch.from_numpy(pd.read_csv(
        os.path.join(extract_path, 'raw', 'node-label.csv.gz'),
        compression='gzip', header=None).values).squeeze().long()

    # Load splits
    print("Loading splits...")
    split_idx = {}
    for split in ['train', 'valid', 'test']:
        split_idx[split] = torch.from_numpy(pd.read_csv(
            os.path.join(extract_path, 'split', 'sales_ranking', f'{split}.csv.gz'),
            compression='gzip', header=None).values.squeeze()).long()

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    # Add masks
    data.train_mask = torch.zeros(x.size(0), dtype=torch.bool)
    data.train_mask[split_idx['train']] = True

    data.val_mask = torch.zeros(x.size(0), dtype=torch.bool)
    data.val_mask[split_idx['valid']] = True

    data.test_mask = torch.zeros(x.size(0), dtype=torch.bool)
    data.test_mask[split_idx['test']] = True

    num_classes = int(y.max().item()) + 1

    print(f"\nNodes: {data.num_nodes:,}")
    print(f"Edges: {data.num_edges:,}")
    print(f"Features: {data.num_features}")
    print(f"Classes: {num_classes}")
    print("="*80 + "\n")

    return data, num_classes


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_examples = 0

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.batch_size
        total_examples += batch.batch_size

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / total_examples


@torch.no_grad()
def evaluate(model, data, split_mask, device, batch_size=4096):
    """Evaluate model"""
    model.eval()

    # Simple batched evaluation
    all_preds = []
    num_nodes = data.num_nodes

    for i in range(0, num_nodes, batch_size):
        end_idx = min(i + batch_size, num_nodes)
        batch_nodes = torch.arange(i, end_idx)

        # Get subgraph
        subset_data = data.subgraph(batch_nodes)
        subset_data = subset_data.to(device)

        out = model(subset_data.x, subset_data.edge_index)
        all_preds.append(out.cpu())

    pred = torch.cat(all_preds, dim=0).argmax(dim=-1)
    correct = (pred[split_mask] == data.y[split_mask]).sum()
    acc = float(correct) / split_mask.sum()

    return acc


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/ogbn_products')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[15, 10, 5])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_every', type=int, default=5)
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data, num_classes = load_data(args.data_dir)

    # Create dataloader
    print("Creating neighbor sampler...")
    train_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        input_nodes=data.train_mask,
        num_workers=args.num_workers,
        shuffle=True
    )

    # Create model (DeepSpeed-ready: no device placement here)
    print("Creating model...")
    model = GraphSAGE(
        input_dim=data.num_features,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_layers=args.num_layers
    )
    model = model.to(device)  # Move to device AFTER creation

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80 + "\n")

    best_val_acc = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        loss = train_epoch(model, train_loader, optimizer, device)
        train_time = time.time() - t0

        # Evaluate
        if epoch % args.eval_every == 0:
            print(f"\nEpoch {epoch:03d} - Evaluating...")
            train_acc = evaluate(model, data, data.train_mask, device)
            val_acc = evaluate(model, data, data.val_mask, device)
            test_acc = evaluate(model, data, data.test_mask, device)

            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f} | "
                  f"Time: {train_time:.1f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pt')
                print(f"  → New best!")
        else:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Time: {train_time:.1f}s")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
