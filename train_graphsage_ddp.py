"""
GraphSAGE training on ogbn-products using PyTorch Geometric with DDP.
Distributed Data Parallel training across multiple GPUs.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import time
from tqdm import tqdm
from math import ceil
import fcntl
import tempfile

# PyTorch Geometric imports
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from torch.nn.parallel import DistributedDataParallel as DDP

# OGB for official dataset loading
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


# ============================================================================
# DDP SETUP FUNCTIONS
# ============================================================================

def setup_ddp():
    """Initialize distributed training from SLURM environment variables.
    
    SLURM sets:
      - SLURM_NTASKS: total number of tasks (= world_size)
      - SLURM_PROCID: global rank of this process
      - SLURM_LOCALID: local rank on this node
    
    We also need MASTER_ADDR and MASTER_PORT set in the SLURM script.
    """
    # Get distributed training parameters from environment
    # Support both torchrun-style and SLURM-style env vars
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', 1)))
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))
    
    # Initialize process group with NCCL backend (optimized for GPU)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',  # Uses MASTER_ADDR and MASTER_PORT from env
        world_size=world_size,
        rank=rank
    )
    
    # Set the CUDA device for this process
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process (rank 0)."""
    return rank == 0


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class GraphSAGE(nn.Module):
    """GraphSAGE model - DDP ready (no device placement in __init__)"""

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
# DATA LOADING WITH DDP
# ============================================================================

def load_data(data_dir, rank, world_size):
    """Load ogbn-products dataset using official OGB loader.
    
    All ranks load the same data (graph is replicated).
    Training indices are split across ranks in create_distributed_loaders().
    """
    if is_main_process(rank):
        print("\n" + "="*80)
        print("LOADING OGBN-PRODUCTS (via OGB)")
        print("="*80)

    # Load dataset using OGB (all ranks)
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

    if is_main_process(rank):
        print(f"\nNodes: {data.num_nodes:,}")
        print(f"Edges: {data.num_edges:,}")
        print(f"Features: {data.num_features}")
        print(f"Classes: {num_classes}")
        print(f"Train: {len(data.train_idx):,} | Val: {len(data.val_idx):,} | Test: {len(data.test_idx):,}")
        print(f"World size: {world_size} GPUs")
        print("="*80 + "\n")

    return data, num_classes, Evaluator(name='ogbn-products')


def create_distributed_loaders(data, args, rank, world_size):
    """Create NeighborLoaders with training indices split across ranks.
    
    Each rank gets a non-overlapping subset of training nodes.
    This provides data parallelism without requiring DistributedSampler.
    """
    # Split training indices across ranks
    train_idx = data.train_idx
    num_train_per_rank = ceil(train_idx.size(0) / world_size)
    
    # Get this rank's portion of training indices
    start_idx = rank * num_train_per_rank
    end_idx = min(start_idx + num_train_per_rank, train_idx.size(0))
    train_idx_split = train_idx[start_idx:end_idx]
    
    if is_main_process(rank):
        print(f"Training samples per rank: ~{len(train_idx_split):,}")
    
    # Create training loader for this rank's subset
    train_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        input_nodes=train_idx_split,  # Only this rank's training nodes
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        shuffle=True,
        drop_last=True,  # Important for DDP to avoid uneven batches causing hangs
    )
    
    return train_loader


# ============================================================================
# TRAINING WITH DDP
# ============================================================================

def train_epoch(model, loader, optimizer, local_rank, rank, accum_steps=1):
    """Train for one epoch with DDP.
    
    Gradients are automatically synchronized across ranks by DDP wrapper.
    """
    model.train()
    total_loss = 0
    total_examples = 0

    optimizer.zero_grad()
    
    # Only show progress bar on rank 0
    pbar = tqdm(loader, desc='Training', disable=not is_main_process(rank))
    
    for i, batch in enumerate(pbar):
        batch = batch.to(local_rank)

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

        if is_main_process(rank):
            pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}'})
    
    # Handle leftover gradients if batches not divisible by accum_steps
    if (i + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / total_examples


@torch.no_grad()
def evaluate(model, data, split_idx, local_rank, num_neighbors, batch_size=16384, num_workers=8):
    """Evaluate model using mini-batch inference with neighbor sampling.
    
    Run on rank 0 only for simplicity.
    Uses the unwrapped model (model.module for DDP).
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
        persistent_workers=True if num_workers > 0 else False,
    )
    
    preds = []
    labels = []
    
    for batch in subgraph_loader:
        batch = batch.to(local_rank)
        out = model(batch.x, batch.edge_index)
        preds.append(out[:batch.batch_size].argmax(dim=-1).cpu())
        labels.append(batch.y[:batch.batch_size].cpu())
    
    pred = torch.cat(preds)
    y_true = torch.cat(labels)
    acc = (pred == y_true).float().mean().item()
    
    return acc


# ============================================================================
# CHECKPOINT AND RESULTS SAVING
# ============================================================================

def save_intermediate_results(row, results_file, rank):
    """Save intermediate results to CSV file, replacing existing row with same ID.
    
    Uses file locking to prevent data races when multiple GPUs write simultaneously.
    If a row with the same ID exists, it is replaced. Otherwise, a new row is appended.
    """
    if not is_main_process(rank):
        return
    
    config_id = row['id']
    lock_file = results_file + '.lock'
    
    try:
        # Acquire lock on the results file
        lock_fd = os.open(lock_file, os.O_CREAT | os.O_WRONLY, 0o666)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # Exclusive lock
        
        try:
            file_exists = os.path.isfile(results_file)
            
            # If file doesn't exist, just write the new row
            if not file_exists:
                with open(results_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    writer.writeheader()
                    writer.writerow(row)
            else:
                # If file exists, read all rows, update/replace the one with same ID
                rows = []
                id_found = False
                
                with open(results_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for existing_row in reader:
                        if existing_row['id'] == str(config_id):
                            # Replace this row with the new one
                            rows.append(row)
                            id_found = True
                        else:
                            rows.append(existing_row)
                
                # If ID wasn't found, append the new row
                if not id_found:
                    rows.append(row)
                
                # Write all rows back using atomic write
                temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(results_file), 
                                                       prefix='.tmp_', suffix='.csv')
                try:
                    with os.fdopen(temp_fd, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=row.keys())
                        writer.writeheader()
                        for r in rows:
                            writer.writerow(r)
                    # Atomic rename
                    os.replace(temp_path, results_file)
                except:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise
        
        finally:
            # Release lock
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    
    except Exception as e:
        print(f"Warning: Failed to save results to {results_file}: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[15, 10, 10, 5, 5],
                        help='Fanout per layer (length must match num_layers)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (number of eval rounds without improvement)')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch = batch_size * accum_steps * world_size')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Path to CSV file for logging results')
    parser.add_argument('--config_id', type=int, default=None,
                        help='Configuration ID for hyperparameter search')
    args = parser.parse_args()
    
    # Validate num_neighbors matches num_layers
    if len(args.num_neighbors) != args.num_layers:
        raise ValueError(f"Length of num_neighbors ({len(args.num_neighbors)}) must match "
                         f"num_layers ({args.num_layers}). Got num_neighbors={args.num_neighbors}")

    # Initialize DDP
    rank, local_rank, world_size = setup_ddp()
    
    if is_main_process(rank):
        print(f"Initialized DDP with {world_size} GPUs")
        print(f"Rank {rank}, Local rank {local_rank}, Device: cuda:{local_rank}")

    # Define checkpoint filename
    if args.config_id is not None:
        checkpoint_path = f'best_model_config_{args.config_id}.pt'
    else:
        checkpoint_path = 'best_model.pt'
    
    # Remove old checkpoint to avoid architecture mismatch
    if is_main_process(rank) and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed old checkpoint: {checkpoint_path}")

    try:
        # Load data (all ranks load the same data)
        data, num_classes, evaluator = load_data(args.data_dir, rank, world_size)

        # Create distributed data loader (each rank gets different training nodes)
        if is_main_process(rank):
            print("Creating distributed neighbor samplers...")
        train_loader = create_distributed_loaders(data, args, rank, world_size)

        # Create model with fixed seed for reproducibility across ranks
        if is_main_process(rank):
            print("Creating model...")
        
        torch.manual_seed(12345)
        model = GraphSAGE(
            input_dim=data.num_features,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers=args.num_layers
        )
        model = model.to(local_rank)
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[local_rank])

        # Scale learning rate with world size (linear scaling rule)
        # If base LR is tuned for batch_size * accum_steps, scale by world_size
        scaled_lr = args.lr * world_size
        
        # Optimizer and LR scheduler with scaled LR
        optimizer = torch.optim.Adam(model.parameters(), lr=scaled_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5 * world_size
        )

        # Training info
        if is_main_process(rank):
            effective_batch = args.batch_size * args.accum_steps * world_size
            print("\n" + "="*80)
            print("TRAINING (DDP)")
            print(f"Batch size: {args.batch_size} x {args.accum_steps} accum x {world_size} GPUs = {effective_batch} effective")
            print(f"Base LR: {args.lr} x {world_size} GPUs = {scaled_lr} scaled LR")
            print("="*80 + "\n")

        best_val_acc = 0
        patience_counter = 0
        eval_num_neighbors = args.num_neighbors
        total_training_time = 0
        final_train_loss = 0
        oom_flag = 0
        results_file = args.results_file if args.results_file else 'results.csv'
        config_id = args.config_id if args.config_id is not None else 0
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats(local_rank)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            # Train (all ranks participate, gradients are synced by DDP)
            loss = train_epoch(model, train_loader, optimizer, local_rank, rank, args.accum_steps)
            scheduler.step()
            train_time = time.time() - t0
            total_training_time += train_time
            final_train_loss = loss
            
            current_lr = scheduler.get_last_lr()[0]

            # Synchronize all ranks before evaluation
            dist.barrier()

            # Evaluate (only rank 0 runs evaluation)
            if epoch % args.eval_every == 0:
                if is_main_process(rank):
                    print(f"\nEpoch {epoch:03d} - Evaluating...")
                    # Use model.module to access the unwrapped model
                    val_acc = evaluate(
                        model.module, data, data.val_idx, local_rank,
                        eval_num_neighbors, num_workers=args.num_workers
                    )

                    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | LR: {current_lr:.2e} | "
                          f"Val: {val_acc:.4f} | Time: {train_time:.1f}s")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                        # Save only the unwrapped model state
                        torch.save(model.module.state_dict(), checkpoint_path)
                        print(f" --- New best!")
                    else:
                        patience_counter += 1
                        print(f" --- No improvement ({patience_counter}/{args.patience})")
                    
                    # Save intermediate results after each evaluation
                    time_per_epoch = total_training_time / epoch
                    num_train_nodes = len(data.train_idx)
                    throughput_nodes_sec = num_train_nodes / time_per_epoch
                    peak_gpu_memory_mb = torch.cuda.max_memory_allocated(local_rank) / (1024 * 1024)
                    
                    intermediate_row = {
                        'id': config_id,
                        'hidden_dim': args.hidden_dim,
                        'num_layers': args.num_layers,
                        'batch_size': args.batch_size,
                        'lr': args.lr,
                        'accum_steps': args.accum_steps,
                        'num_neighbors': ' '.join(map(str, args.num_neighbors)),
                        'num_epochs': epoch,
                        'best_val_acc': best_val_acc,
                        'test_acc': 0.0,  # Test acc not available until end
                        'training_time_s': total_training_time,
                        'time_per_epoch_sec': time_per_epoch,
                        'throughput_nodes_sec': throughput_nodes_sec,
                        'peak_gpu_memory_mb': peak_gpu_memory_mb,
                        'oom_flag': oom_flag,
                        'train_loss_final': final_train_loss
                    }
                    save_intermediate_results(intermediate_row, results_file, rank)
                
                # Broadcast early stopping decision to all ranks
                should_stop = torch.tensor([patience_counter >= args.patience], device=local_rank)
                dist.broadcast(should_stop, src=0)
                
                if should_stop.item():
                    if is_main_process(rank):
                        print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
            else:
                if is_main_process(rank):
                    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | LR: {current_lr:.2e} | Time: {train_time:.1f}s")
            
            # Synchronize at end of epoch
            dist.barrier()

        # Final evaluation on test set (rank 0 only)
        if is_main_process(rank):
            print("\n" + "="*80)
            print("FINAL EVALUATION")
            print("="*80)
            
            # Load best model if checkpoint exists
            if os.path.exists(checkpoint_path):
                model.module.load_state_dict(torch.load(checkpoint_path))
                print(f"Loaded best model from {checkpoint_path}")
            else:
                print(f"Warning: No checkpoint found at {checkpoint_path}, using current model")
            
            test_acc = evaluate(
                model.module, data, data.test_idx, local_rank,
                eval_num_neighbors, num_workers=args.num_workers
            )
            
            # Calculate metrics
            time_per_epoch = total_training_time / epoch
            num_train_nodes = len(data.train_idx)
            throughput_nodes_sec = num_train_nodes / time_per_epoch
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated(local_rank) / (1024 * 1024)
            
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            print(f"Test accuracy: {test_acc:.4f}")
            print(f"Time per epoch: {time_per_epoch:.2f}s")
            print(f"Throughput: {throughput_nodes_sec:.0f} nodes/sec")
            print(f"Peak GPU memory: {peak_gpu_memory_mb:.2f} MB")
            print("="*80)

            # Final results row (update test_acc from intermediate checkpoint)
            final_row = {
                'id': config_id,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'accum_steps': args.accum_steps,
                'num_neighbors': ' '.join(map(str, args.num_neighbors)),
                'num_epochs': epoch,
                'best_val_acc': best_val_acc,
                'test_acc': test_acc,
                'training_time_s': total_training_time,
                'time_per_epoch_sec': time_per_epoch,
                'throughput_nodes_sec': throughput_nodes_sec,
                'peak_gpu_memory_mb': peak_gpu_memory_mb,
                'oom_flag': oom_flag,
                'train_loss_final': final_train_loss
            }
            
            # Overwrite the intermediate results with final test_acc
            save_intermediate_results(final_row, results_file, rank)

    except torch.cuda.OutOfMemoryError as e:
        # Handle OOM errors
        if is_main_process(rank):
            print(f"\n{'='*80}")
            print("OUT OF MEMORY ERROR")
            print(f"{'='*80}")
            print(f"Error: {e}")
            
            results_file = args.results_file if args.results_file else 'results.csv'
            config_id = args.config_id if args.config_id is not None else 0
            
            row = {
                'id': config_id,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'accum_steps': args.accum_steps,
                'num_neighbors': ' '.join(map(str, args.num_neighbors)),
                'num_epochs': 0,
                'best_val_acc': 0.0,
                'test_acc': 0.0,
                'training_time_s': 0.0,
                'time_per_epoch_sec': 0.0,
                'throughput_nodes_sec': 0.0,
                'peak_gpu_memory_mb': 0.0,
                'oom_flag': 1,
                'train_loss_final': 0.0
            }
            
            save_intermediate_results(row, results_file, rank)
            print(f"OOM error logged to {results_file}")
        
        cleanup_ddp()
        raise
    
    finally:
        # Always cleanup DDP
        cleanup_ddp()


if __name__ == "__main__":
    main()
