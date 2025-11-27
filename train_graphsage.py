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
from torch_geometric.loader import NeighborLoader

# OGB for official dataset loading
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class GraphSAGE(nn.Module):
    """GraphSAGE model - DeepSpeed ready (no device placement in __init__)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5, dropout=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()  # LayerNorm as recommended for ogbn-products

        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.lns.append(nn.LayerNorm(hidden_dim))

        # Output layer (no norm, no activation, no dropout - clean output)
        self.convs.append(SAGEConv(hidden_dim, output_dim))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.lns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


# ============================================================================
# DATA LOADING (using official OGB loader)
# ============================================================================

def load_data(data_dir):
    """Load ogbn-products dataset using official OGB loader.
    
    Benefits over manual loading:
    - Correct graph format guaranteed
    - Benchmark-standard splits
    - Automatic caching
    - No CSV parsing dependencies
    """
    print("\n" + "="*80)
    print("LOADING OGBN-PRODUCTS (via OGB)")
    print("="*80)

    # Load dataset using OGB
    dataset = PygNodePropPredDataset(name='ogbn-products', root=data_dir)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    
    # Flatten labels from [N, 1] to [N]
    data.y = data.y.squeeze()
    
    # Store split indices on data object for convenience
    data.train_idx = split_idx['train']
    data.val_idx = split_idx['valid']
    data.test_idx = split_idx['test']
    
    num_classes = dataset.num_classes

    print(f"\nNodes: {data.num_nodes:,}")
    print(f"Edges: {data.num_edges:,}")
    print(f"Features: {data.num_features}")
    print(f"Classes: {num_classes}")
    print(f"Train: {len(data.train_idx):,} | Val: {len(data.val_idx):,} | Test: {len(data.test_idx):,}")
    print("="*80 + "\n")

    return data, num_classes, Evaluator(name='ogbn-products')


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, device, accum_steps=1):
    """Train for one epoch with optional gradient accumulation.
    
    Args:
        accum_steps: Number of micro-batches to accumulate before optimizer step.
                     Effective batch size = batch_size * accum_steps
    """
    model.train()
    total_loss = 0
    total_examples = 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc='Training')
    for i, batch in enumerate(pbar):
        batch = batch.to(device)

        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
        
        # Scale loss for gradient accumulation
        loss = loss / accum_steps
        loss.backward()

        # Step optimizer every accum_steps batches
        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps * batch.batch_size
        total_examples += batch.batch_size

        pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}'})
    
    # Handle leftover gradients if batches not divisible by accum_steps
    if (i + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / total_examples


@torch.no_grad()
def evaluate(model, data, split_idx, device, num_neighbors, batch_size=16384, num_workers=8):
    """Evaluate model using mini-batch inference with neighbor sampling.
    
    Only evaluates nodes in the given split (not all 2.4M nodes).
    Uses larger batch size and fewer neighbors for faster inference.
    """
    model.eval()
    
    # Use smaller fanout for faster eval (half of training fanout)
    eval_fanout = [max(1, n // 2) for n in num_neighbors]
    
    # Only evaluate nodes in this split
    subgraph_loader = NeighborLoader(
        data,
        num_neighbors=eval_fanout,
        batch_size=batch_size,
        input_nodes=split_idx,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    preds = []
    labels = []
    
    for batch in subgraph_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        preds.append(out[:batch.batch_size].argmax(dim=-1).cpu())
        labels.append(batch.y[:batch.batch_size].cpu())
    
    pred = torch.cat(preds)
    y_true = torch.cat(labels)
    acc = (pred == y_true).float().mean().item()
    
    return acc


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[15, 10, 10, 5, 5],
                        help='Fanout per layer (length must match num_layers)')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (number of eval rounds without improvement)')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch = batch_size * accum_steps')
    args = parser.parse_args()
    
    # Validate num_neighbors matches num_layers
    if len(args.num_neighbors) != args.num_layers:
        raise ValueError(f"Length of num_neighbors ({len(args.num_neighbors)}) must match "
                         f"num_layers ({args.num_layers}). Got num_neighbors={args.num_neighbors}")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data using OGB
    data, num_classes, evaluator = load_data(args.data_dir)

    # Create dataloader
    print("Creating neighbor sampler...")
    train_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        input_nodes=data.train_idx,
        num_workers=args.num_workers,
        pin_memory=True, # speed up host to device transfer
        persistent_workers=True, # keep workers alive for multiple epochs
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

    # Optimizer and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print(f"Batch size: {args.batch_size} x {args.accum_steps} accum = {args.batch_size * args.accum_steps} effective")
    print("="*80 + "\n")

    best_val_acc = 0
    patience_counter = 0
    
    # Use training fanout for inference (eval function will halve it)
    eval_num_neighbors = args.num_neighbors

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        loss = train_epoch(model, train_loader, optimizer, device, args.accum_steps)
        scheduler.step()  # Update LR
        train_time = time.time() - t0
        
        current_lr = scheduler.get_last_lr()[0]

        # Evaluate (mini-batch inference with neighbor sampling)
        if epoch % args.eval_every == 0:
            print(f"\nEpoch {epoch:03d} - Evaluating...")
            train_acc = evaluate(model, data, data.train_idx, device, eval_num_neighbors, num_workers=args.num_workers) 
            val_acc = evaluate(model, data, data.val_idx, device, eval_num_neighbors, num_workers=args.num_workers)

            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | LR: {current_lr:.2e} | "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                  f"Time: {train_time:.1f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
                print(f" --- New best!")
            else:
                patience_counter += 1
                print(f" --- No improvement ({patience_counter}/{args.patience})")
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
        else:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | LR: {current_lr:.2e} | Time: {train_time:.1f}s")

    # Final evaluation on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    test_acc = evaluate(model, data, data.test_idx, device, eval_num_neighbors, num_workers=args.num_workers)
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
