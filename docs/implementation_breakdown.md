# Mainrun Implementation Breakdown

## Overview

This document breaks down the optimization strategy into actionable epics, tasks, and subtasks for systematic implementation. Each epic builds upon previous work, with clear success criteria and risk mitigation.

**Target**: Reduce validation loss from baseline 1.7533 to < 1.70 (primary), < 1.60 (stretch)
**Timeline**: 7 days
**Strategy**: Incremental implementation with validation at each stage

---

## EPIC 1: FOUNDATIONAL TRAINING FIXES (Days 1-2)
**Goal**: Fix critical training configuration issues to beat baseline
**Priority**: HIGHEST - These changes alone should achieve primary goal
**Expected Impact**: 0.10-0.20 validation loss reduction

### Task 1.1: Replace SGD with AdamW Optimizer
**Rationale**: SGD cannot handle transformer gradient heterogeneity
**Files**: `mainrun/train.py` (lines 265-266)

- [ ] **Subtask 1.1.1**: Create backup of current train.py
  - [ ] [ ] Copy `train.py` to `train_baseline.py` 
  - [ ] [ ] Ensure baseline can be reproduced
  
- [ ] **Subtask 1.1.2**: Implement AdamW optimizer
  ```python
  # Replace lines 265-266
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=3e-4,  # Down from 6e-3
      weight_decay=0.1,  # Up from 0.0
      betas=(0.9, 0.999),
      eps=1e-8
  )
  ```
  - [ ] [ ] Update hyperparameters class with new defaults
  
- [ ] **Subtask 1.1.3**: Run initial validation test
  - [ ] [ ] Execute single epoch training
  - [ ] [ ] Verify loss decreases and no errors
  - [ ] [ ] Compare first epoch loss vs baseline
  
- [ ] **Subtask 1.1.4**: Document initial results
  - [ ] [ ] Record hyperparameters used
  - [ ] [ ] Note any immediate improvements

### Task 1.2: Implement Learning Rate Warmup + Cosine Decay
**Rationale**: Prevent early training instability and enable higher peak LR
**Files**: `mainrun/train.py` (scheduler section)

- [ ] **Subtask 1.2.1**: Calculate warmup parameters
  ```python
  warmup_steps = max_steps // 10  # ~94 steps
  ```
  
- [ ] **Subtask 1.2.2**: Implement warmup scheduler
  ```python
  warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
      opt, 
      start_factor=0.01,  # Start at 1% of peak LR
      end_factor=1.0,
      total_iters=warmup_steps
  )
  ```
  
- [ ] **Subtask 1.2.3**: Implement cosine decay
  ```python
  cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      opt,
      T_max=max_steps - warmup_steps,
      eta_min=3e-5  # 10% of peak LR
  )
  ```
  
- [ ] **Subtask 1.2.4**: Chain schedulers
  ```python
  from torch.optim.lr_scheduler import SequentialLR
  scheduler = SequentialLR(
      opt,
      schedulers=[warmup_scheduler, cosine_scheduler],
      milestones=[warmup_steps]
  )
  ```
  
- [ ] **Subtask 1.2.5**: Validate schedule behavior
  - [ ] [ ] Plot LR curve over training steps
  - [ ] [ ] Ensure smooth warmup and decay

### Task 1.3: Execute Full Training Run
**Goal**: Validate foundational fixes beat baseline
**Success Criteria**: Final validation loss < 1.70

- [ ] **Subtask 1.3.1**: Run complete 7-epoch training
  - [ ] [ ] Execute `task train`
  - [ ] [ ] Monitor for stability/divergence
  - [ ] [ ] Save logs with timestamp
  
- [ ] **Subtask 1.3.2**: Analyze results
  - [ ] [ ] Compare final validation loss to baseline (1.7533)
  - [ ] [ ] Examine training curves for stability
  - [ ] [ ] Identify any remaining issues
  
- [ ] **Subtask 1.3.3**: Document Epic 1 results
  - [ ] [ ] Record final validation loss
  - [ ] [ ] Note training stability
  - [ ] [ ] Save model checkpoint if improved

---

## EPIC 2: ARCHITECTURAL MODERNIZATION (Days 3-4)
**Goal**: Replace standard components with modern alternatives
**Priority**: MEDIUM-HIGH - Efficiency and performance gains
**Expected Impact**: 0.02-0.05 validation loss reduction + speed improvements

### Task 2.1: Replace LayerNorm with RMSNorm
**Rationale**: 7-64% speed improvement with comparable performance
**Files**: `mainrun/train.py` (Block class, GPT class)

- [ ] **Subtask 2.1.1**: Implement RMSNorm class
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
  
- [ ] **Subtask 2.1.2**: Replace LayerNorm instances
  - [ ] [ ] Update Block class: `self.ln1 = RMSNorm(cfg.d_model)`
  - [ ] [ ] Update Block class: `self.ln2 = RMSNorm(cfg.d_model)` 
  - [ ] [ ] Update GPT class: `self.ln_f = RMSNorm(cfg.d_model)`
  
- [ ] **Subtask 2.1.3**: Validate forward pass
  - [ ] [ ] Run single forward pass and compare shapes
  - [ ] [ ] Ensure gradients flow properly
  - [ ] [ ] Check for NaN/inf values
  
- [ ] **Subtask 2.1.4**: Benchmark speed improvement
  - [ ] [ ] Time forward pass before/after change
  - [ ] [ ] Measure memory usage if possible

### Task 2.2: Implement Rotary Position Embeddings (RoPE)
**Rationale**: Parameter-free relative positioning with better generalization
**Files**: `mainrun/train.py` (CausalSelfAttention, GPT classes)

- [ ] **Subtask 2.2.1**: Implement RoPE class
  ```python
  class RotaryEmbedding(nn.Module):
      def __init__(self, dim, max_position_embeddings=2048, base=10000):
          super().__init__()
          self.dim = dim
          self.max_position_embeddings = max_position_embeddings
          self.base = base
          
          inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
          self.register_buffer("inv_freq", inv_freq)
          
      def forward(self, x, seq_len=None):
          # Implementation of rotary position embeddings
  ```
  
- [ ] **Subtask 2.2.2**: Remove learned positional embeddings
  - [ ] [ ] Remove `self.pos_emb` from GPT class
  - [ ] [ ] Update forward pass to not add positional embeddings
  
- [ ] **Subtask 2.2.3**: Integrate RoPE into attention
  - [ ] [ ] Modify CausalSelfAttention to apply rotations to q,k
  - [ ] [ ] Ensure correct position encoding application
  
- [ ] **Subtask 2.2.4**: Validate attention mechanism
  - [ ] [ ] Check attention weights make sense
  - [ ] [ ] Verify causal mask still works
  - [ ] [ ] Test on sample sequences

### Task 2.3: Improve Weight Initialization
**Rationale**: He initialization better for GELU activation
**Files**: `mainrun/train.py` (_init_weights method)

- [ ] **Subtask 2.3.1**: Update initialization method
  ```python
  @staticmethod
  def _init_weights(module):
      if isinstance(module, nn.Linear):
          nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
          if module.bias is not None:
              nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
  ```
  
- [ ] **Subtask 2.3.2**: Apply to all modules
  - [ ] [ ] Ensure initialization runs on all parameters
  - [ ] [ ] Verify RMSNorm weights initialized correctly
  
- [ ] **Subtask 2.3.3**: Validate gradient flow
  - [ ] [ ] Check initial gradient magnitudes
  - [ ] [ ] Ensure no vanishing/exploding gradients

### Task 2.4: Execute Full Training Run
**Goal**: Validate architectural improvements
**Success Criteria**: Maintain or improve upon Epic 1 results

- [ ] **Subtask 2.4.1**: Run complete 7-epoch training
  - [ ] Use Epic 1 hyperparameters as baseline
  - [ ] Monitor training stability
  
- [ ] **Subtask 2.4.2**: Compare performance
  - [ ] Validation loss vs Epic 1
  - [ ] Training speed improvements
  - [ ] Memory usage changes
  
- [ ] **Subtask 2.4.3**: Document Epic 2 results
  - [ ] Record architectural changes impact
  - [ ] Note any performance improvements

---

## EPIC 3: HYPERPARAMETER OPTIMIZATION (Days 5-6)
**Goal**: Fine-tune hyperparameters for optimal performance
**Priority**: MEDIUM - Squeeze out final performance gains
**Expected Impact**: 0.01-0.05 additional validation loss reduction

### Task 3.1: Learning Rate Grid Search
**Goal**: Find optimal learning rate for final configuration

- [ ] **Subtask 3.1.1**: Define search grid
  - [ ] Test values: [1e-4, 3e-4, 5e-4]
  - [ ] Keep all other hyperparameters fixed
  
- [ ] **Subtask 3.1.2**: Execute training runs
  - [ ] Run 7 epochs for each LR value
  - [ ] Use identical seeds for fair comparison
  
- [ ] **Subtask 3.1.3**: Analyze results
  - [ ] Plot validation curves for all LR values
  - [ ] Identify best performing learning rate
  - [ ] Check for overfitting vs underfitting

### Task 3.2: Regularization Tuning
**Goal**: Balance overfitting prevention with model capacity

- [ ] **Subtask 3.2.1**: Weight decay sweep
  - [ ] Test values: [0.05, 0.1, 0.15] 
  - [ ] Use best LR from Task 3.1
  
- [ ] **Subtask 3.2.2**: Dropout rate sweep  
  - [ ] Test values: [0.05, 0.1, 0.15, 0.2]
  - [ ] Apply consistently across all dropout locations
  
- [ ] **Subtask 3.2.3**: Joint optimization
  - [ ] Test best combinations from individual sweeps
  - [ ] Select configuration with lowest validation loss

### Task 3.3: Batch Size Optimization
**Goal**: Maximize hardware utilization and gradient quality

- [ ] **Subtask 3.3.1**: Find maximum batch size
  - [ ] Test increasing batch sizes until OOM
  - [ ] Note memory constraints
  
- [ ] **Subtask 3.3.2**: Implement gradient accumulation
  ```python
  # If batch size needs to be reduced
  accumulation_steps = desired_batch_size // actual_batch_size
  if step % accumulation_steps == 0:
      opt.step()
      opt.zero_grad()
  ```
  
- [ ] **Subtask 3.3.3**: Apply learning rate scaling
  - [ ] If batch size changed significantly from 64
  - [ ] Use linear scaling rule: `new_lr = base_lr * (new_batch / old_batch)`

### Task 3.4: Final Optimization Run
**Goal**: Achieve stretch goal of < 1.60 validation loss

- [ ] **Subtask 3.4.1**: Execute with best hyperparameters
  - [ ] Use optimal configuration from all sweeps
  - [ ] Run multiple seeds for statistical significance
  
- [ ] **Subtask 3.4.2**: Validate results
  - [ ] Ensure consistent improvements across runs
  - [ ] Check for statistical significance
  
- [ ] **Subtask 3.4.3**: Document final configuration
  - [ ] Record all optimal hyperparameters
  - [ ] Save best model checkpoint

---

## EPIC 4: DOCUMENTATION & SUBMISSION (Day 7)
**Goal**: Create comprehensive report and submit solution
**Priority**: REQUIRED - Assessment requirement
**Deliverable**: `mainrun/report.pdf`

### Task 4.1: Generate Training Visualizations
**Goal**: Create compelling visual evidence of improvements

- [ ] **Subtask 4.1.1**: Loss curve comparisons
  - [ ] Plot baseline vs optimized training curves
  - [ ] Highlight key improvement milestones
  - [ ] Show validation loss progression
  
- [ ] **Subtask 4.1.2**: Learning rate schedule visualization
  - [ ] Plot LR over training steps
  - [ ] Show warmup and cosine decay phases
  
- [ ] **Subtask 4.1.3**: Hyperparameter sensitivity analysis
  - [ ] Show impact of different LR values
  - [ ] Demonstrate regularization effects

### Task 4.2: Write Comprehensive Report
**Goal**: Document methodology, results, and insights

- [ ] **Subtask 4.2.1**: Executive summary
  - [ ] Baseline vs final performance
  - [ ] Key changes implemented
  - [ ] Validation loss improvements
  
- [ ] **Subtask 4.2.2**: Methodology section
  - [ ] Explain each optimization with rationale
  - [ ] Reference research supporting decisions
  - [ ] Detail implementation approach
  
- [ ] **Subtask 4.2.3**: Results analysis
  - [ ] Present final validation loss (target < 1.60)
  - [ ] Show training stability improvements
  - [ ] Discuss architectural benefits
  
- [ ] **Subtask 4.2.4**: Insights and future work
  - [ ] What worked best and why
  - [ ] What didn't work as expected
  - [ ] Potential further improvements

### Task 4.3: Prepare Submission
**Goal**: Ensure submission meets all requirements

- [ ] **Subtask 4.3.1**: Final code cleanup
  - [ ] Remove debug prints and unused code
  - [ ] Ensure consistent formatting
  - [ ] Add docstrings to new classes
  
- [ ] **Subtask 4.3.2**: Export report to PDF
  - [ ] Place `report.pdf` in `mainrun/` folder
  - [ ] Ensure all visualizations render correctly
  
- [ ] **Subtask 4.3.3**: Execute submission
  - [ ] Run final `task train` to generate latest logs
  - [ ] Execute `task submit` command
  - [ ] Verify submission completed successfully

---

## Risk Mitigation Strategy

### Technical Risks
1. **Model divergence**: Implement gradient clipping and monitor for NaN/inf
2. **Memory constraints**: Use gradient accumulation if batch size limited
3. **Implementation bugs**: Test each component individually before integration

### Process Risks  
1. **Time management**: Prioritize Epic 1 (highest impact) first
2. **Baseline comparison**: Always maintain ability to reproduce baseline
3. **Checkpoint failures**: Save intermediate models after each epic

### Success Criteria by Epic
- **Epic 1**: Validation loss < 1.70 (beat baseline)
- **Epic 2**: Maintain Epic 1 performance with speed gains
- **Epic 3**: Achieve stretch goal < 1.60 if possible
- **Epic 4**: Complete, professional documentation

This implementation plan provides a systematic approach to achieving significant improvements while managing risk through incremental validation at each stage.