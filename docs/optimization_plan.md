# Mainrun Optimization Plan

## Executive Summary

This comprehensive optimization plan addresses the suboptimal baseline validation loss of **1.7533** by implementing evidence-based improvements in three prioritized phases. The baseline model suffers from fundamental training configuration issues, particularly the use of SGD with high learning rate and no warmup, which severely hampers convergence.

## Current Baseline Analysis
- **Final validation loss**: 1.7533 (after 7 epochs, 938 total steps)
- **Target**: Beat baseline by achieving significantly lower validation loss
- **Model**: GPT-2 style transformer (6 layers, 8 heads, 512 d_model)
- **Critical Issues**: SGD optimizer, 6e-3 learning rate, no warmup, no weight decay

## Part I: Foundational Training Dynamics (HIGHEST IMPACT)

### 1.1 Replace SGD with AdamW Optimizer
**Rationale**: The baseline's use of SGD is fundamentally mismatched for transformers due to "block heterogeneity" - gradient scales vary dramatically across parameter blocks (embeddings vs upper layers). SGD's single global learning rate cannot handle this effectively.

**Implementation**:
```python
# Replace current optimizer
opt = torch.optim.AdamW(
    model.parameters(), 
    lr=3e-4,  # Much lower than 6e-3
    weight_decay=0.1,  # Add regularization
    betas=(0.9, 0.999),
    eps=1e-8
)
```

**Expected Impact**: HIGH - AdamW's adaptive per-parameter learning rates directly solve the Hessian heterogeneity problem that SGD cannot handle.

### 1.2 Implement Learning Rate Warmup + Cosine Decay
**Rationale**: Cold starts with high learning rates cause catastrophic instability. Warmup prevents destructive early updates and enables higher effective peak learning rates.

**Implementation**:
```python
# Warmup for ~10% of total steps (94 steps out of 938)
warmup_steps = max_steps // 10
scheduler = torch.optim.lr_scheduler.LinearLR(
    opt, start_factor=0.01, total_iters=warmup_steps
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=max_steps - warmup_steps, eta_min=3e-5
)
# Use SequentialLR to chain them
```

**Expected Impact**: HIGH - Enables stable training and access to higher peak learning rates.

### 1.3 Optimize Learning Rate for AdamW
**Current**: 6e-3 (orders of magnitude too high for AdamW)
**Recommended**: 3e-4 (empirically validated range: 1e-5 to 5e-4)

## Part II: Architectural Modernization (MEDIUM-HIGH IMPACT)

### 2.1 Replace Learned Positional Embeddings with RoPE
**Rationale**: RoPE provides parameter-free relative positional encoding that generalizes better and injects geometric understanding directly into attention.

**Implementation**:
```python
# Remove pos_emb parameter, add RoPE to attention
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        # Implementation details...
        
# Apply in CausalSelfAttention forward pass
```

**Expected Impact**: MEDIUM-HIGH - Better relative position understanding, improved generalization.

### 2.2 Replace LayerNorm with RMSNorm
**Rationale**: RMSNorm eliminates re-centering (mean subtraction), keeping only re-scaling. Provides 7-64% speed improvement with comparable performance.

**Implementation**:
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)
```

**Expected Impact**: MEDIUM - Faster training enables more experiments within time budget.

### 2.3 Improve Weight Initialization
**Current**: Default PyTorch initialization
**Recommended**: He initialization for GELU activation

**Implementation**:
```python
def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

## Part III: Regularization & Fine-tuning (MEDIUM IMPACT)

### 3.1 Strategic Regularization
**Weight Decay**: 0.1 (standard for transformers, implemented in AdamW)
**Dropout**: Maintain 0.1 but apply strategically:
- Embedding dropout
- Attention dropout
- Residual dropout

### 3.2 Batch Size Optimization
**Current**: 64
**Strategy**: Use largest batch size that fits in memory, consider gradient accumulation if needed
**Learning Rate Scaling**: If batch size changes significantly, apply linear scaling rule

## Implementation Priority & Expected Impact

| Priority | Change | Expected Validation Loss Reduction | Implementation Effort |
|----------|--------|-----------------------------------|---------------------|
| 1 | AdamW Optimizer | 0.05-0.10 | Low |
| 2 | Learning Rate (6e-3 → 3e-4) | 0.03-0.07 | Trivial |
| 3 | LR Warmup + Cosine Decay | 0.02-0.05 | Medium |
| 4 | Weight Decay (0.0 → 0.1) | 0.01-0.03 | Trivial |
| 5 | RoPE Implementation | 0.01-0.04 | High |
| 6 | RMSNorm | 0.00-0.01 (speed gain) | Medium |

**Total Expected Improvement**: 0.12-0.30 reduction in validation loss
**Target**: From 1.7533 to **1.45-1.63** (well below baseline)

## Detailed Implementation Plan

### Phase 1: Core Training Fixes (Day 1)
1. Replace SGD with AdamW
2. Set learning rate to 3e-4
3. Add weight decay of 0.1
4. Implement basic warmup schedule

### Phase 2: Advanced Scheduling (Day 2)
1. Implement proper warmup + cosine decay
2. Fine-tune warmup steps (5-15% of total)
3. Set appropriate eta_min for cosine decay

### Phase 3: Architecture Modernization (Days 3-4)
1. Implement RMSNorm
2. Add RoPE (most complex change)
3. Improve weight initialization

### Phase 4: Validation & Tuning (Day 5+)
1. Run experiments with different learning rates
2. Tune weight decay and dropout
3. Optimize batch size if needed

## Risk Mitigation

1. **Implement incrementally**: Make one change at a time to isolate impact
2. **Monitor training curves**: Watch for instability or overfitting
3. **Keep baseline comparison**: Always compare against 1.7533 baseline
4. **Hyperparameter sensitivity**: Test ranges around recommended values

## Success Metrics

- **Primary**: Validation loss < 1.70 (beat baseline)
- **Stretch**: Validation loss < 1.60 (significant improvement)
- **Training stability**: Smooth convergence without divergence
- **Reproducibility**: Consistent results across multiple runs

This plan is grounded in extensive research and empirical evidence from the transformer literature, providing a high-confidence path to substantial performance improvements.