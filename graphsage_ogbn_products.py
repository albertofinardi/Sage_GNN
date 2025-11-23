import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import urllib.request
import zipfile
import pandas as pd
import time
import sys
from collections import defaultdict
import random

print("[INIT] ============================================================")
print("[INIT] GraphSAGE - Starting up...")
print(f"[INIT] PyTorch version: {torch.__version__}")
print(f"[INIT] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INIT] CUDA version: {torch.version.cuda}")
print("[INIT] ============================================================")


class SAGEConv(nn.Module):
    """GraphSAGE convolution - vectorized."""
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__()
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)
        
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        
        # Aggregate neighbors using scatter
        out = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        degree = torch.zeros(num_nodes, device=x.device, dtype=torch.float)
        degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        degree = degree.clamp(min=1)
        out.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        out = out / degree.unsqueeze(1)
        
        return self.lin_neigh(out) + self.lin_self(x)

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(GraphSAGE, self).__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        x = F.normalize(x, p=2, dim=1)
        return x

def download_ogbn_products(data_dir='./data/ogbn_products'):
    print(f"[Download] Checking data directory: {data_dir}")
    
    
    os.makedirs(data_dir, exist_ok=True)
    extract_path = os.path.join(data_dir, 'products')
    
    if os.path.exists(extract_path):
        print(f"[Download] Dataset already exists")
        
        return extract_path
    
    zip_path = os.path.join(data_dir, 'products.zip')
    url = 'https://snap.stanford.edu/ogb/data/nodeproppred/products.zip'
    print(f"[Download] Downloading from OGB...")
    
    
    urllib.request.urlretrieve(url, zip_path)
    print(f"[Download] Extracting...")
    
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print(f"[Download] Complete")
    
    
    return extract_path

def load_ogbn_products(data_dir='./data/ogbn_products'):
    print("\n[Load] ========== Loading Dataset ==========")
    
    
    extract_path = download_ogbn_products(data_dir)
    
    print("[Load] Step 1/4: Node features...")
    
    t0 = time.time()
    x = torch.from_numpy(pd.read_csv(
        os.path.join(extract_path, 'raw', 'node-feat.csv.gz'),
        compression='gzip', header=None).values).float()
    print(f"[Load]   {x.shape[0]:,} nodes, {x.shape[1]} features ({time.time()-t0:.1f}s)")
    
    
    print("[Load] Step 2/4: Edges...")
    
    t0 = time.time()
    edge_index = torch.from_numpy(pd.read_csv(
        os.path.join(extract_path, 'raw', 'edge.csv.gz'),
        compression='gzip', header=None).values.T).long()
    print(f"[Load]   {edge_index.shape[1]:,} edges ({time.time()-t0:.1f}s)")
    
    
    print("[Load] Step 3/4: Labels...")
    
    t0 = time.time()
    y = torch.from_numpy(pd.read_csv(
        os.path.join(extract_path, 'raw', 'node-label.csv.gz'),
        compression='gzip', header=None).values).squeeze().long()
    num_classes = int(y.max().item()) + 1
    print(f"[Load]   {num_classes} classes ({time.time()-t0:.1f}s)")
    
    
    print("[Load] Step 4/4: Splits...")
    
    split_idx = {}
    for split in ['train', 'valid', 'test']:
        split_idx[split] = torch.from_numpy(pd.read_csv(
            os.path.join(extract_path, 'split', 'sales_ranking', f'{split}.csv.gz'),
            compression='gzip', header=None).values.squeeze()).long()
    print(f"[Load]   Train: {len(split_idx['train']):,}, Valid: {len(split_idx['valid']):,}, Test: {len(split_idx['test']):,}")
    print("[Load] ========================================\n")
    
    
    return x, edge_index, y, split_idx, num_classes

class NeighborSampler:
    """Sample K-hop neighbors for mini-batch training."""
    def __init__(self, edge_index, node_idx, batch_size, num_neighbors):
        self.node_idx = node_idx.tolist() if torch.is_tensor(node_idx) else list(node_idx)
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        
        # Build adjacency list
        print(f"[Sampler] Building adjacency list...")
        
        t0 = time.time()
        self.adj = defaultdict(list)
        for src, dst in edge_index.t().tolist():
            self.adj[dst].append(src)
        print(f"[Sampler] Adjacency list built in {time.time()-t0:.1f}s")
        
    
    def sample(self, nodes, num_samples):
        """Sample neighbors for nodes."""
        sampled_nodes = set(nodes)
        
        for node in nodes:
            neighbors = self.adj[node]
            if len(neighbors) <= num_samples or num_samples < 0:
                sampled_nodes.update(neighbors)
            else:
                sampled_nodes.update(random.sample(neighbors, num_samples))
        
        return list(sampled_nodes)
    
    def __iter__(self):
        indices = self.node_idx.copy()
        random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_nodes = indices[i:i + self.batch_size]
            
            # Sample 1-hop neighbors only (memory efficient)
            all_nodes = self.sample(batch_nodes, self.num_neighbors[0])
            
            # Map to local indices
            node_map = {n: i for i, n in enumerate(all_nodes)}
            batch_local = [node_map[n] for n in batch_nodes]
            
            # Extract subgraph edges
            subgraph_edges = []
            for node in all_nodes:
                if node in self.adj:
                    for neighbor in self.adj[node]:
                        if neighbor in node_map:
                            subgraph_edges.append([node_map[neighbor], node_map[node]])
            
            if len(subgraph_edges) > 0:
                edge_index = torch.tensor(subgraph_edges, dtype=torch.long).t()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            yield all_nodes, batch_local, edge_index

@torch.no_grad()
def evaluate(model, x, edge_index, y, split_idx, device, batch_size=4096):
    """Evaluate in mini-batches to avoid OOM."""
    model.eval()
    
    print("    [Eval] Running evaluation in batches to avoid OOM...")
    
    
    # Process in batches
    all_preds = []
    num_nodes = x.size(0)
    edge_index_gpu = edge_index.to(device)
    
    for i in range(0, num_nodes, batch_size):
        end_idx = min(i + batch_size, num_nodes)
        
        # Get subgraph for this batch
        batch_nodes = list(range(i, end_idx))
        batch_x = x[batch_nodes].to(device)
        
        # Create dummy edges for this batch (no message passing for eval)
        out = model(batch_x, torch.zeros((2, 0), dtype=torch.long, device=device))
        all_preds.append(out.cpu())
        
        if (i // batch_size) % 100 == 0:
            print(f"      Processed {end_idx:,}/{num_nodes:,} nodes...")
            
    
    pred = torch.cat(all_preds, dim=0).argmax(dim=-1)
    
    train_acc = (pred[split_idx['train']] == y[split_idx['train']]).float().mean().item()
    valid_acc = (pred[split_idx['valid']] == y[split_idx['valid']]).float().mean().item()
    test_acc = (pred[split_idx['test']] == y[split_idx['test']]).float().mean().item()
    
    return train_acc, valid_acc, test_acc

def train_graphsage():    
    # Load data
    x, edge_index, y, split_idx, num_classes = load_ogbn_products()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Setup] Device: {device}")
    if torch.cuda.is_available():
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Setup] Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print()
    
    
    # Model
    print("[Model] Creating model...")
    
    model = GraphSAGE(
        input_dim=x.size(1),
        hidden_dim=256,
        output_dim=num_classes,
        num_layers=3,
        dropout=0.5
    ).to(device)
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create neighbor sampler
    batch_size = 512
    num_neighbors = [10]  # Sample 10 neighbors per node (1-hop only)
    sampler = NeighborSampler(edge_index, split_idx['train'], batch_size, num_neighbors)
    
    print(f"[Config] Batch size: {batch_size}")
    print(f"[Config] Neighbor samples: {num_neighbors}")
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")
    
    
    best_valid_acc = 0
    patience_counter = 0
    patience = 50
    
    for epoch in range(1, 501):
        t0 = time.time()
        
        model.train()
        total_loss = 0
        total_examples = 0
        batch_count = 0
        
        for all_nodes, batch_local, subgraph_edges in sampler:
            # Transfer subgraph to GPU
            batch_x = x[all_nodes].to(device)
            batch_y = y[[all_nodes[i] for i in batch_local]].to(device)
            subgraph_edges = subgraph_edges.to(device)
            
            optimizer.zero_grad()
            
            # Forward on subgraph only
            out = model(batch_x, subgraph_edges)
            loss = loss_fn(out[batch_local], batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item() * len(batch_local)
            total_examples += len(batch_local)
            batch_count += 1
        
        avg_loss = total_loss / total_examples
        train_time = time.time() - t0
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            print(f"[Epoch {epoch:03d}] Evaluating...")
            
            
            eval_start = time.time()
            train_acc, valid_acc, test_acc = evaluate(
                model, x, edge_index, y, split_idx, device
            )
            eval_time = time.time() - eval_start
            
            epoch_total = time.time() - t0
            
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                  f"Train: {train_acc:.4f} | Valid: {valid_acc:.4f} | Test: {test_acc:.4f} | "
                  f"Time: {epoch_total:.1f}s")
            
            if torch.cuda.is_available():
                print(f"  GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
                print(f"  NEW BEST")
            else:
                patience_counter += 10
            
            print()
            
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                
                break
        else:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Time: {train_time:.1f}s")
            
    
    # Final
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    
    model.load_state_dict(torch.load('best_model.pt'))
    train_acc, valid_acc, test_acc = evaluate(
        model, x, edge_index, y, split_idx, device
    )
    
    print(f"Best Valid: {best_valid_acc:.4f} ({best_valid_acc*100:.2f}%)")
    print(f"Final Test: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 80)
    

if __name__ == "__main__":
    train_graphsage()