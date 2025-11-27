# ogbn-products Dataset

## Overview
Amazon product co-purchasing network with 2.4M nodes (products) and 61M edges (co-purchased pairs).

**Task**: Predict product category (47 classes) from graph structure and node features.

## File Structure

```
products/
├── raw/
│   ├── node-feat.csv.gz    # [2.4M × 100] Bag-of-words features (PCA-reduced)
│   ├── edge.csv.gz         # [61M × 2] Edge list (source, target)
│   └── node-label.csv.gz   # [2.4M × 1] Category labels (0-46)
│
└── split/sales_ranking/
    ├── train.csv.gz        # ~196K indices (top 8% by popularity)
    ├── valid.csv.gz        # ~39K indices (next 2%)
    └── test.csv.gz         # ~2.2M indices (remaining 90%)
```

## Split Strategy

Based on **sales ranking** (not random):

| Split | % of nodes | Description |
|-------|-----------|-------------|
| Train | 8% | Most popular products |
| Valid | 2% | Medium popularity |
| Test | 90% | Long-tail products |

## Why This Split is Challenging

1. **Distribution shift**: Popular products have different characteristics than long-tail
2. **Fewer connections**: Less popular products have sparser neighborhoods
3. **Real-world scenario**: Mirrors labeling popular items first, predicting the rest

## Data Leakage Considerations

- **Features**: Node features are intrinsic (product descriptions) — no leakage
- **Edges**: Co-purchasing is transductive (edges exist at train time) — standard for GNNs
- **Labels**: Only train labels should be used during training
- **Evaluation**: Don't tune hyperparameters on test set
