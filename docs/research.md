

# **A Prioritized Optimization Plan for Transformer-Based Text Generation Models**

## **Executive Summary: A Roadmap to Optimal Performance**

This report presents a comprehensive, prioritized optimization plan for a GPT-2 style transformer model tasked with text generation. The primary objective is to systematically reduce the final validation loss from a suboptimal baseline of 1.7533. The baseline model's performance is severely hampered by foundational choices in its training configuration, particularly the use of Stochastic Gradient Descent (SGD) with a high learning rate and no warmup period, which represents a significant deviation from established best practices for transformer training. These choices create an unstable optimization landscape, preventing the model from converging to a desirable minimum.

The optimization strategy is structured in three prioritized parts, designed to deliver the most significant performance gains first:

1. **Part I: Foundational Optimization of Training Dynamics.** This section addresses the most critical issues—the optimizer and learning rate schedule—to establish a stable and effective training dynamic. These changes are expected to yield the largest initial reduction in validation loss.  
2. **Part II: Architectural Modernization for Enhanced Representation and Efficiency.** This part introduces modern, empirically superior architectural components, such as advanced positional embeddings and normalization layers, that improve the model's representational capacity and computational efficiency.  
3. **Part III: Fine-Tuning and Regularization for Generalization.** The final section details the implementation and tuning of regularization techniques to combat overfitting and refines hyperparameters like batch size to further improve generalization on the held-out validation set.

The following tables summarize the baseline configuration and the prioritized recommendations that form the core of this report.

**Table 1: Baseline Model Configuration and Performance**

| Component | Baseline Setting | Deficiency |
| :---- | :---- | :---- |
| **Architecture** | 6-layer, 8-head GPT-2 Style | Standard, but can be improved with modern components. |
| **Optimizer** | Stochastic Gradient Descent (SGD) | Suboptimal for transformer architectures; cannot handle heterogeneous gradients. |
| **Learning Rate** | 6×10−3 | Excessively high, especially for an adaptive optimizer; likely causes instability. |
| **LR Scheduler** | Cosine Annealing (no warmup) | Lacks the critical warmup phase necessary for stable transformer training. |
| **Weight Decay** | 0.0 | No regularization, leading to a high risk of overfitting. |
| **Positional Embeddings** | Learned Absolute | Parameter-inefficient and prone to poor generalization on novel sequence lengths. |
| **Normalization Layer** | Standard Layer Normalization | Computationally suboptimal compared to modern alternatives. |
| **Validation Loss** | 1.7533 | High, indicating significant room for improvement. |

**Table 2: Prioritized Optimization Recommendations and Expected Impact**

| Priority | Recommendation | Rationale | Expected Impact on Validation Loss |
| :---- | :---- | :---- | :---- |
| 1 | Switch to **AdamW** Optimizer | Addresses the fundamental issue of Hessian heterogeneity in transformers and provides superior decoupled weight decay. | High |
| 2 | Implement **LR Warmup & Cosine Decay** | Ensures training stability, prevents early divergence, and enables the use of a higher, more effective peak learning rate. | High |
| 3 | Set Appropriate **Peak LR for AdamW** | Corrects the excessively high baseline learning rate to a range empirically proven to be effective for AdamW on transformers. | High |
| 4 | Introduce **Weight Decay & Dropout** | Implements crucial regularization to combat overfitting and improve generalization on the validation set. | Medium-High |
| 5 | Replace Learned Embeddings with **RoPE** | Modernizes positional encoding to be parameter-free, inherently relative, and better at generalization. | Medium |
| 6 | Replace LayerNorm with **RMSNorm** | Improves computational efficiency with comparable performance, enabling more rapid experimentation. | Low-Medium |
| 7 | Tune **Batch Size** | Optimizes the trade-off between gradient accuracy and regularization noise for final performance refinement. | Low |

By systematically implementing these recommendations, it is possible to transform the underperforming baseline into a robust, high-performing text generation model that achieves a significantly lower validation loss.

## **Part I: Foundational Optimization of Training Dynamics**

The most critical flaws of the baseline model lie in its training dynamics. The choice of optimizer, learning rate, and schedule are misaligned with the known properties of transformer architectures. Correcting this foundation is the highest-leverage intervention and must be performed before any other changes are considered. These initial steps are expected to yield the most substantial improvement in model performance.

### **1.1 Rectifying the Optimizer: Transitioning from SGD to AdamW**

The primary reason for the baseline's poor performance is the selection of Stochastic Gradient Descent (SGD) as the optimizer. While SGD is a robust and memory-efficient optimizer for architectures like Convolutional Neural Networks (CNNs), it is fundamentally ill-suited for the complex and challenging optimization landscape of transformers.1

A key reason for this mismatch lies in a property of transformers known as "block heterogeneity." Research has demonstrated that the curvature of the loss surface, which is mathematically described by the Hessian matrix, varies dramatically across different parameter blocks within the model.2 For instance, the gradients associated with the initial embedding layer can be orders of magnitude larger than those in the upper layers.4 SGD applies a single, global learning rate to all parameters, which is incapable of effectively navigating this varied landscape. A learning rate that is appropriate for one block may be excessively large for another, causing instability and divergence, or too small for a third, leading to prohibitively slow convergence.

Adaptive optimizers like Adam, and specifically its modern variant AdamW, are designed to solve this exact problem. AdamW maintains per-parameter adaptive learning rates derived from exponential moving averages of the first and second moments of the gradients (the mean and uncentered variance).1 This mechanism effectively normalizes the gradient update for each parameter, applying a tailored, coordinate-wise learning rate that accounts for the block heterogeneity.2 This intrinsic ability to handle disparate gradient scales is why Adam-family optimizers consistently and significantly outperform SGD on transformer-based tasks.1 The choice of AdamW is therefore not merely a hyperparameter tweak; it is a structural solution to a fundamental property of the transformer loss landscape.

Furthermore, the "W" in AdamW signifies a crucial improvement in how regularization is handled. The baseline model's complete lack of weight decay is a form of under-regularization that must be corrected. When introducing this technique, AdamW is the superior choice over standard Adam combined with L2 regularization. AdamW implements a *decoupled* weight decay, where the weight penalty is applied directly to the weights *after* the gradient-based update step.5 In standard Adam, the L2 penalty is coupled with the gradient itself, which interferes with the adaptive momentum and variance calculations, potentially leading to suboptimal updates and poorer generalization. By decoupling these two components, AdamW ensures that the regularization effect is more stable and consistent, making it the de-facto standard optimizer for training modern large-scale models.7

While AdamW does require more GPU memory than SGD (approximately 16 bytes per parameter versus 12 bytes for SGD with momentum), the substantial performance and stability gains for transformer models far outweigh this modest additional cost, especially for a model of the specified size.4

**Table 3: Comparison of Optimizer Characteristics (SGD vs. AdamW)**

| Characteristic | Stochastic Gradient Descent (SGD) | AdamW (Adam with Decoupled Weight Decay) |
| :---- | :---- | :---- |
| **Learning Rate Mechanism** | Global (a single LR for all parameters). | Adaptive (a per-parameter LR adjusted by gradient moments). |
| **Memory per Parameter** | \~12 bytes (with momentum). | \~16 bytes. |
| **Convergence on Transformers** | Slower and often converges to a suboptimal solution.1 | Faster and empirically superior, achieving lower loss.2 |
| **Handling of Heterogeneous Gradients** | Poorly suited; cannot adapt to the different gradient scales across parameter blocks, leading to instability or slow training.2 | Well-suited; normalizes updates based on each parameter's gradient history, effectively handling Hessian block heterogeneity.1 |
| **Weight Decay Implementation** | Requires a manual L2 penalty to be added to the loss function, which is not part of the optimizer itself. | Implements a decoupled weight decay, applying it directly to the weights, which leads to more stable training and better generalization.6 |

### **1.2 Stabilizing Convergence: Implementing a Learning Rate Warmup and Cosine Decay Schedule**

The baseline model's training process begins with a high learning rate from the very first step, a practice known as a "cold start." This is highly problematic. At the beginning of training, the model's weights are randomly initialized, and the initial gradients can be large, noisy, and uninformative. Applying a large learning rate immediately can lead to catastrophic instability, where the weight updates are so large and erratic that they push the model into a poor region of the loss landscape from which it cannot recover, especially within a fixed training budget of only seven epochs.10

A learning rate warmup phase is a non-negotiable best practice for training transformers and is essential for achieving stable convergence.12 This technique involves starting with a very small learning rate (often 0\) and linearly increasing it to the target peak learning rate over a specified number of initial training steps, known as the "warmup steps".12 This gradual increase serves two critical functions. First, it allows the model to take small, controlled steps at the outset, preventing the destructive, oversized updates that can occur with a cold start.10 Second, and crucially for an adaptive optimizer like AdamW, it gives the momentum (

m) and variance (v) accumulators time to build a reliable, stable estimate of the gradient statistics before the learning rate becomes large. This mitigates the high variance in the adaptive learning rate that can otherwise occur in early training when the optimizer has seen too few samples.12

The primary benefit of this warmup period is that it enables the use of a much higher peak learning rate than would otherwise be stable.12 It functions as a dynamic preconditioner for the optimization problem. By taking small initial steps, it guides the model away from poorly-conditioned regions of the loss landscape (areas of high "sharpness" or curvature) and toward flatter regions where larger, more aggressive steps can be taken safely and effectively. Without warmup, one is forced to use a tiny, potentially suboptimal learning rate to avoid divergence. With warmup, a more aggressive and effective peak learning rate becomes accessible, which is key to achieving faster convergence and a lower final loss.

The number of warmup steps is a hyperparameter, but a common and effective heuristic is to set it to 5-10% of the total number of training steps.16 For a fixed 7-epoch run, this provides a concrete and reliable starting point for implementation.

Following the warmup phase, the learning rate should be gradually decreased, or annealed, to allow the model to fine-tune its parameters and settle into a good minimum. The baseline's use of a cosine annealing schedule is a sound choice, but its effectiveness is crippled by the lack of a preceding warmup. A combined schedule that consists of a linear warmup followed by a cosine decay to a small final value (e.g., 10% of the peak learning rate) is a robust, state-of-the-art strategy.17 This schedule smoothly decreases the learning rate following the shape of a cosine curve, a method that has been shown to be highly effective in practice for a wide range of deep learning tasks.11

### **1.3 Selecting an Optimal Learning Rate for AdamW**

The optimizer, learning rate, and schedule form an inseparable system; optimizing one in isolation is ineffective. The baseline fails because all three components are misconfigured *relative to each other*. The specified learning rate of 6×10−3 might be plausible in some SGD contexts, but it is orders of magnitude too high for AdamW. The adaptive nature of AdamW, which scales updates based on gradient history, means it generally requires and performs best with much smaller learning rates.19

Based on extensive empirical evidence from the research community, a suitable starting range for the peak learning rate for AdamW when training small-to-medium transformer models is between 1×10−5 and 5×10−4.13 A commonly cited and robust starting point within this range is

3×10−4.20 The excessively high learning rate of the baseline is a primary contributor to its poor validation score, as it likely causes the optimizer to repeatedly overshoot minima and fail to converge properly.11

Therefore, the first experiment after implementing the AdamW optimizer and the warmup/decay schedule should be to test a peak learning rate of **3×10−4**. This single change, in conjunction with the new optimizer and schedule, is expected to produce the most dramatic improvement over the baseline. Further tuning can then explore nearby values, such as 1×10−4 and 5×10−4, to find the optimal point for this specific dataset and model.

## **Part II: Architectural Modernization for Enhanced Representation and Efficiency**

After establishing a stable training foundation by correcting the optimizer and learning rate schedule, the next priority is to upgrade the model's architecture. This involves replacing standard components from the original GPT-2 design with modern alternatives that have been empirically proven to enhance representational capacity, improve generalization, and increase computational efficiency.

### **2.1 Enhancing Positional Representation: Replacing Learned Embeddings with RoPE**

The baseline model uses learned absolute positional embeddings, a technique where a unique vector is learned for each position in the sequence up to a maximum length. This approach has two significant weaknesses. First, it requires a large embedding matrix of size (maximum\_sequence\_length × hidden\_dimension), which adds a considerable number of parameters to the model. Second, and more critically, it struggles to generalize to sequence lengths longer than those seen during training, as the model has no learned representation for these novel positions.22

Rotary Position Embeddings (RoPE) offer a more elegant, powerful, and parameter-free solution that has become the standard in most modern high-performing LLMs, including LLaMA and PaLM.24 Instead of

*adding* a positional vector to the token embedding, RoPE *applies a rotation* to the query and key vectors within the self-attention mechanism.22 The angle of this rotation is a deterministic function of the token's absolute position in the sequence.26

The profound benefit of this approach becomes apparent during the dot-product attention calculation. The dot product between two rotated vectors is mathematically equivalent to a function of their original, unrotated vectors and their *relative* positional distance (m−n).26 This means the attention mechanism inherently becomes sensitive to the relative positions of tokens without ever needing to explicitly learn this concept. This property leads to significantly better generalization and performance, particularly on tasks that require understanding long-range dependencies and complex syntactic structures.23

RoPE is more than just a different type of embedding; it is an architectural prior that injects the geometric concept of relative distance directly into the self-attention mechanism. Learned embeddings force the model to expend its limited capacity on discovering the fundamental concept of sequence order from scratch. In contrast, RoPE provides this information for free through its mathematical formulation. This allows the model to dedicate its parameters to learning more complex linguistic patterns, resulting in better sample efficiency and a lower final validation loss.

### **2.2 Improving Normalization Efficiency: A Case for RMSNorm over LayerNorm**

Normalization layers are critical for stabilizing the training of deep transformers. They ensure that the inputs to each layer remain in a well-behaved, predictable range, which in turn helps maintain a stable and consistent flow of gradients during backpropagation.29 The baseline model uses standard Layer Normalization (LayerNorm), which normalizes the activations within a layer by performing two operations: re-centering (subtracting the mean) and re-scaling (dividing by the standard deviation).31

Root Mean Square Normalization (RMSNorm) is a simplification of LayerNorm that has gained widespread adoption in modern LLMs like T5 and LLaMA.29 RMSNorm simplifies the process by removing the re-centering step and only performing re-scaling, normalizing the activations by their root mean square value.29 The core hypothesis behind RMSNorm is that the re-centering invariance provided by LayerNorm is largely unnecessary for the performance of transformers, while the re-scaling invariance is the critical component for stabilization.33 This hypothesis is strongly supported by empirical results, which show that models using RMSNorm achieve performance comparable to those using LayerNorm.29

The primary advantage of RMSNorm is its computational efficiency. By eliminating the mean calculation, RMSNorm is simpler and faster, with studies reporting reductions in running time of 7% to 64%, depending on the specific model and hardware.33 While this efficiency gain may not directly alter the final loss value of a single converged run, it accelerates the training process. This increased speed allows for more experiments, such as comprehensive hyperparameter sweeps, to be conducted within a fixed time budget. This increased experimental velocity is an indirect but powerful pathway to discovering a better-performing model configuration. Adopting RMSNorm is therefore a low-risk, high-efficiency change that aligns the model with modern best practices.

### **2.3 Ensuring Stable Gradient Flow: Correct Weight Initialization**

Proper weight initialization is a subtle but foundational aspect of training deep neural networks. It is crucial for ensuring that signals and gradients can flow effectively through the network's many layers at the start of training. Poor initialization can lead to activation values that are either too small (vanishing gradients) or too large (exploding gradients), both of which can stall or destabilize the learning process.35

Two of the most common initialization schemes are Xavier (or Glorot) initialization and He initialization. The choice between them depends directly on the type of activation function used in the network:

* **Xavier (Glorot) Initialization:** This method was designed to maintain activation variance when using symmetric, zero-centered activation functions like the hyperbolic tangent (tanh).37 It typically samples weights from a distribution with a variance of  
  1/nin​, where nin​ is the number of input connections to the neuron.  
* **He Initialization:** This method was specifically developed for the Rectified Linear Unit (ReLU) and its variants, such as the Gaussian Error Linear Unit (GELU) used in GPT-2. Since ReLU-like functions set all negative inputs to zero, they effectively halve the variance of their inputs on average. He initialization compensates for this by doubling the variance of the weight distribution to 2/nin​, ensuring that the activation variance is properly maintained across layers.35

Given that the GPT-2 style model architecture employs the GELU activation function, which is a smooth approximation of ReLU, **He initialization is the theoretically correct and practically superior choice**. Using the older Xavier initialization would likely lead to a decaying signal variance as the signal propagates through the network, resulting in slower and less stable training. This change ensures the network starts in a healthy, well-conditioned state, ready for effective optimization.

### **2.4 Note on Computational Acceleration: The Role of Flash Attention**

While the previous recommendations modify the mathematical definition of the model, Flash Attention is an implementation-level optimization that computes the *exact same* attention output, but does so much faster and with significantly less memory.40

The primary bottleneck in the standard self-attention mechanism is not the number of floating-point operations (FLOPs) but rather the memory bandwidth required to read and write the large, intermediate N×N attention score matrix to and from the GPU's relatively slow High-Bandwidth Memory (HBM).42 Flash Attention is an "IO-aware" algorithm that restructures the computation to minimize these costly memory transfers. It uses techniques like tiling to load small blocks of query, key, and value vectors into the GPU's much faster on-chip SRAM. It then computes the attention output for that block entirely within SRAM and writes only the final, smaller output vector back to HBM, completely avoiding the materialization of the full attention matrix.43

This restructuring results in significant speedups (2-4x is common) and reduces the memory requirement of the attention mechanism from being quadratic in the sequence length, O(N2), to linear, O(N).41 This efficiency gain is an indirect but powerful enabler of better model performance. It allows for training with a larger batch size, which can improve gradient stability, or training on longer sequences, which enhances the model's ability to capture long-range dependencies. Both of these can lead to a lower final validation loss. For many modern deep learning frameworks, such as PyTorch 2.0+ and libraries like Hugging Face Transformers, Flash Attention can often be enabled with a single configuration flag (e.g.,

attn\_implementation="flash\_attention\_2"), making it a highly impactful and easily accessible optimization.46

## **Part III: Fine-Tuning and Regularization for Generalization**

With a stable and modern architecture in place, the final step in the optimization process is to apply and tune regularization techniques. These methods are designed to prevent the model from overfitting to the training data, thereby improving its ability to generalize to the unseen validation set and achieve the lowest possible validation loss.

### **3.1 Strategic Regularization: Introducing Weight Decay and Dropout**

The baseline model's complete absence of explicit regularization makes it highly susceptible to overfitting, which is a likely contributor to its high validation loss. Two complementary techniques, weight decay and dropout, should be introduced to address this deficiency.

**Weight Decay:** As established in Part I, the decoupled weight decay implemented in the AdamW optimizer is the preferred method for transformers. A non-zero weight\_decay parameter should be set in the optimizer configuration. A common and effective starting value for transformer models is **0.1**, with a typical tuning range between 0.01 and 0.1.8 Weight decay acts as a form of L2 regularization, penalizing large weight values and encouraging the model to find simpler, smoother solutions that are less likely to memorize noise in the training data and thus generalize better.49

**Dropout:** Dropout is a vital regularization technique that prevents the co-adaptation of neurons by randomly setting a fraction of activations to zero during each forward pass in training.50 This forces the network to learn more robust and redundant representations, as it cannot rely on any single neuron or feature being present. In the standard transformer architecture, dropout is typically applied at several key locations to maximize its effectiveness 52:

1. **Embedding Dropout:** Applied to the output of the combined token and positional embeddings layer.  
2. **Sub-layer Dropout:** Applied to the output of each sub-layer (i.e., after the self-attention block and after the feed-forward network block), before the output is added to the residual connection.  
3. **Attention Dropout:** Applied directly to the attention probability matrix (the scores after scaling but before the softmax operation). This encourages the model to distribute its attention more broadly rather than focusing too heavily on a few tokens.

A common starting dropout rate (p) for all these locations is **0.1**.52 This value provides a moderate level of regularization. Higher rates (e.g., 0.3 or more) can be overly aggressive for smaller models or datasets and risk causing underfitting, while lower rates may not provide sufficient regularization.54 The dropout rate is a key hyperparameter that should be tuned based on the validation loss.

Weight decay and dropout are not mutually exclusive; they are complementary. Weight decay acts as a global, continuous constraint on the magnitude of the model's parameters, while dropout acts as a stochastic, structural regularizer that alters the network's functional form at each training step. Applying both techniques in tandem addresses different potential failure modes of generalization and is a standard practice for robustly training transformer models.

### **3.2 A Note on Batch Size and Its Interplay with Learning Rate**

The choice of batch size involves a fundamental trade-off between computational efficiency, training speed, and model generalization.55

* **Large Batch Sizes:** These lead to more accurate and stable gradient estimates because they average over more samples. This allows for better hardware utilization, particularly on modern GPUs, resulting in faster training time per epoch. However, the smooth gradients can cause the optimizer to converge to "sharp" minima in the loss landscape, which have been shown to generalize more poorly to unseen data.58  
* **Small Batch Sizes:** These introduce significant noise into the gradient updates. This noise can act as a form of regularization, helping the optimizer to escape poor local minima and find "flatter" minima that tend to generalize better.58 However, the noisy updates can slow down convergence, and the poor hardware utilization can make training much slower overall.

Given the fixed 7-epoch training constraint, maximizing hardware utilization to process as much data as efficiently as possible is important. A pragmatic approach is to **start with the largest batch size that comfortably fits into available GPU memory**.61 If memory constraints are a significant issue,

**gradient accumulation** is an effective technique to simulate a larger effective batch size without the corresponding memory overhead. This involves performing several forward and backward passes with smaller batches and accumulating the gradients before performing a single weight update.59

It is also important to note the strong interplay between batch size and learning rate. If the batch size is changed significantly, the learning rate often needs to be adjusted to compensate. A common (though not universally perfect) heuristic is the "linear scaling rule," which suggests that if the batch size is multiplied by a factor of k, the learning rate should also be multiplied by k (or in some cases, k​) to maintain similar training dynamics.61

### **3.3 Considerations for Architectural Scaling: Layers, Heads, and Hidden Dimensions**

While the user's model architecture (6 layers, 8 heads) is fixed for this optimization task, understanding the principles of transformer scaling provides valuable context for interpreting model behavior and planning future work.

The performance of transformer models generally improves with more parameters, but how those parameters are allocated—between depth (number of layers) and width (hidden dimension, number of heads)—matters. For a fixed parameter budget, research suggests that increasing depth is often more beneficial than increasing width.62 Deeper models can learn more complex, hierarchical representations of the data, where each layer builds upon the abstractions learned by the previous one. However, this effect is subject to diminishing returns. Very deep but narrow models can suffer from training instability and may underperform compared to more balanced architectures.62 There appears to be a "just deep enough" principle, where a model should have sufficient depth to capture hierarchical features, after which adding parameters to its width may be more effective.

The multi-head attention mechanism allows the model to jointly attend to information from different representation subspaces at different positions.65 However, extensive research has shown that many attention heads within a fully trained model are often redundant and can be pruned away with minimal to no impact on performance.66 This indicates that the representational capacity of some heads is either not utilized or is duplicative of other heads. This suggests that the baseline's 8 heads per layer is a reasonable and likely sufficient number for a 6-layer model. Simply increasing the number of heads is unlikely to yield significant benefits without a corresponding increase in other model dimensions, training data, and computational budget.

## **Conclusion: A Synthesized Action Plan**

This report has outlined a systematic, evidence-based, and prioritized plan to significantly improve the performance of the baseline text generation model. The strategy moves logically from foundational corrections to the training process, through architectural modernization, to fine-grained regularization and tuning. By following these steps, a substantial reduction in the validation loss from the starting point of 1.7533 is highly probable.

The recommended action plan is as follows:

1. **Replace SGD with the AdamW optimizer** to properly handle the transformer's optimization landscape.  
2. **Implement a linear warmup and cosine decay learning rate schedule**, using approximately 10% of total training steps for the warmup phase to ensure stable convergence.  
3. **Set the peak learning rate to an appropriate value for AdamW**, starting with 3×10−4 and tuning from there.  
4. **Introduce regularization by setting a non-zero weight decay** (e.g., 0.1) in the AdamW optimizer and **applying dropout** (e.g., 0.1) at standard locations within the transformer architecture.  
5. (Recommended for further gains) **Replace the learned absolute positional embeddings with Rotary Position Embeddings (RoPE)** to improve the model's handling of relative positions and its generalization capabilities.  
6. (Recommended for efficiency) **Replace standard Layer Normalization with RMSNorm** to accelerate training and enable more rapid experimentation.  
7. Once a stable and well-performing configuration is established, begin a more focused **hyperparameter search** on the peak learning rate, weight decay, and dropout rate, using the validation loss as the guiding metric.

The following table provides a final, actionable summary of the key hyperparameters and their recommended ranges for initial tuning experiments.

**Table 4: Recommended Hyperparameter Ranges for Initial Tuning**

| Hyperparameter | Recommended Starting Value | Recommended Tuning Range | Justification / Key Sources |
| :---- | :---- | :---- | :---- |
| **Peak Learning Rate (AdamW)** | 3×10−4 | \[1×10−5, 6×10−4\] | Empirically validated range for transformers.19 |
| **Weight Decay** | 0.1 | \[0.0, 0.1\] | Standard practice for regularizing large models; AdamW's default is often too low.8 |
| **Dropout Rate** | 0.1 | \[0.0, 0.3\] | A common starting point that balances regularization with preventing underfitting.52 |
| **Warmup Steps (% of total)** | 10% | \[5%, 15%\] | A robust heuristic for ensuring training stability.16 |
| **Batch Size** | Largest that fits in memory | Constrained by hardware; adjust LR accordingly. | Maximizes hardware utilization and gradient stability.61 |

#### **Works cited**

1. Understanding Why Adam Outperforms SGD: Gradient Heterogeneity in Transformers, accessed on August 26, 2025, [https://arxiv.org/html/2502.00213v1](https://arxiv.org/html/2502.00213v1)  
2. Why Transformers Need Adam: A Hessian Perspective \- NIPS, accessed on August 26, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/ee0e45ff4de76cbfdf07015a7839f339-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/ee0e45ff4de76cbfdf07015a7839f339-Paper-Conference.pdf)  
3. Stochastic Gradient Descent (SGD) and Adam | by Hey Amit | Data Scientist's Diary, accessed on August 26, 2025, [https://medium.com/data-scientists-diary/stochastic-gradient-descent-sgd-and-adam-4fe496ef1bbf](https://medium.com/data-scientists-diary/stochastic-gradient-descent-sgd-and-adam-4fe496ef1bbf)  
4. How to Fine-Tune Vision Models with SGD \- arXiv, accessed on August 26, 2025, [https://arxiv.org/pdf/2211.09359](https://arxiv.org/pdf/2211.09359)  
5. AdamW Optimizer in PyTorch Tutorial \- DataCamp, accessed on August 26, 2025, [https://www.datacamp.com/tutorial/adamw-optimizer-in-pytorch](https://www.datacamp.com/tutorial/adamw-optimizer-in-pytorch)  
6. Adam vs. AdamW: Understanding Weight Decay and Its Impact on Model Performance, accessed on August 26, 2025, [https://yassin01.medium.com/adam-vs-adamw-understanding-weight-decay-and-its-impact-on-model-performance-b7414f0af8a1](https://yassin01.medium.com/adam-vs-adamw-understanding-weight-decay-and-its-impact-on-model-performance-b7414f0af8a1)  
7. What are the key differences between Adam and AdamW optimizers in the context of transformer-based language models? \- Massed Compute, accessed on August 26, 2025, [https://massedcompute.com/faq-answers/?question=What%20are%20the%20key%20differences%20between%20Adam%20and%20AdamW%20optimizers%20in%20the%20context%20of%20transformer-based%20language%20models?](https://massedcompute.com/faq-answers/?question=What+are+the+key+differences+between+Adam+and+AdamW+optimizers+in+the+context+of+transformer-based+language+models?)  
8. How to set AdamW's weight decay as you scale model and dataset size \- arXiv, accessed on August 26, 2025, [https://arxiv.org/html/2405.13698v1](https://arxiv.org/html/2405.13698v1)  
9. No More Adam: Learning Rate Scaling at Initialization is All You Need \- Hugging Face, accessed on August 26, 2025, [https://huggingface.co/papers/2412.11768](https://huggingface.co/papers/2412.11768)  
10. Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training \- arXiv, accessed on August 26, 2025, [https://arxiv.org/html/2410.23922v1](https://arxiv.org/html/2410.23922v1)  
11. 12.11. Learning Rate Scheduling — Dive into Deep Learning 1.0.3 documentation, accessed on August 26, 2025, [https://d2l.ai/chapter\_optimization/lr-scheduler.html](https://d2l.ai/chapter_optimization/lr-scheduler.html)  
12. Why Warmup the Learning Rate? Underlying Mechanisms and Improvements \- arXiv, accessed on August 26, 2025, [https://arxiv.org/html/2406.09405v1](https://arxiv.org/html/2406.09405v1)  
13. \[R\] Tips on training Transformers : r/MachineLearning \- Reddit, accessed on August 26, 2025, [https://www.reddit.com/r/MachineLearning/comments/z088fo/r\_tips\_on\_training\_transformers/](https://www.reddit.com/r/MachineLearning/comments/z088fo/r_tips_on_training_transformers/)  
14. In the context of Deep Learning, what is training warmup steps, accessed on August 26, 2025, [https://datascience.stackexchange.com/questions/55991/in-the-context-of-deep-learning-what-is-training-warmup-steps](https://datascience.stackexchange.com/questions/55991/in-the-context-of-deep-learning-what-is-training-warmup-steps)  
15. FAQ | Machine Learning \- Google for Developers, accessed on August 26, 2025, [https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/faq](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/faq)  
16. HuggingFace's linear scheduler with warmup parameters \- Stack Overflow, accessed on August 26, 2025, [https://stackoverflow.com/questions/73054136/huggingfaces-linear-scheduler-with-warmup-parameters](https://stackoverflow.com/questions/73054136/huggingfaces-linear-scheduler-with-warmup-parameters)  
17. Optimizer Schedules — Optax documentation \- Read the Docs, accessed on August 26, 2025, [https://optax.readthedocs.io/en/latest/api/optimizer\_schedules.html](https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html)  
18. Cosine Learning rate decay \- Sebastian Correa \- Medium, accessed on August 26, 2025, [https://scorrea92.medium.com/cosine-learning-rate-decay-e8b50aa455b](https://scorrea92.medium.com/cosine-learning-rate-decay-e8b50aa455b)  
19. What learning rate value is considered safe? : r/learnmachinelearning \- Reddit, accessed on August 26, 2025, [https://www.reddit.com/r/learnmachinelearning/comments/14hntl6/what\_learning\_rate\_value\_is\_considered\_safe/](https://www.reddit.com/r/learnmachinelearning/comments/14hntl6/what_learning_rate_value_is_considered_safe/)  
20. Is it good learning rate for Adam method? \- Stack Overflow, accessed on August 26, 2025, [https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method](https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method)  
21. \[D\] Good studies on the effects of different training "tricks" like learning rate scheduler (warmup/decay), weight decay, dropout, batch-sizes, momentum, etc.? \- Reddit, accessed on August 26, 2025, [https://www.reddit.com/r/MachineLearning/comments/1fihdrd/d\_good\_studies\_on\_the\_effects\_of\_different/](https://www.reddit.com/r/MachineLearning/comments/1fihdrd/d_good_studies_on_the_effects_of_different/)  
22. Rotary Positional Embeddings: A Detailed Look and Comprehensive Understanding | by azhar \- Medium, accessed on August 26, 2025, [https://medium.com/ai-insights-cobet/rotary-positional-embeddings-a-detailed-look-and-comprehensive-understanding-4ff66a874d83](https://medium.com/ai-insights-cobet/rotary-positional-embeddings-a-detailed-look-and-comprehensive-understanding-4ff66a874d83)  
23. Compared to the baseline, how much does RoPE improve LLMs? \- ResearchGate, accessed on August 26, 2025, [https://www.researchgate.net/post/Compared\_to\_the\_baseline\_how\_much\_does\_RoPE\_improve\_LLMs](https://www.researchgate.net/post/Compared_to_the_baseline_how_much_does_RoPE_improve_LLMs)  
24. Extending the RoPE \- EleutherAI Blog, accessed on August 26, 2025, [https://blog.eleuther.ai/yarn/](https://blog.eleuther.ai/yarn/)  
25. Inside RoPE: Rotary Magic into Position Embeddings \- LearnOpenCV, accessed on August 26, 2025, [https://learnopencv.com/rope-position-embeddings/](https://learnopencv.com/rope-position-embeddings/)  
26. Rotary Positional Embeddings (RoPE) \- labml.ai, accessed on August 26, 2025, [https://nn.labml.ai/transformers/rope/index.html](https://nn.labml.ai/transformers/rope/index.html)  
27. Rotary Positional Embeddings (RoPE) \- The Large Language Model Playbook, accessed on August 26, 2025, [https://cyrilzakka.github.io/llm-playbook/nested/rot-pos-embed.html?utm\_source=hnblogs.substack.com](https://cyrilzakka.github.io/llm-playbook/nested/rot-pos-embed.html?utm_source=hnblogs.substack.com)  
28. Benchmarking Rotary Position Embeddings for Automatic Speech Recognition \- arXiv, accessed on August 26, 2025, [https://arxiv.org/html/2501.06051v1](https://arxiv.org/html/2501.06051v1)  
29. Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and, accessed on August 26, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2023/file/8f1bacee31caf990a4f08d84f0ccb322-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/8f1bacee31caf990a4f08d84f0ccb322-Paper-Conference.pdf)  
30. Normalization Techniques in Transformer-Based LLMs: LayerNorm, RMSNorm, and Beyond, accessed on August 26, 2025, [https://sushant-kumar.com/blog/normalization-in-transformer-based-llms](https://sushant-kumar.com/blog/normalization-in-transformer-based-llms)  
31. bzhangGo/rmsnorm: Root Mean Square Layer Normalization \- GitHub, accessed on August 26, 2025, [https://github.com/bzhangGo/rmsnorm](https://github.com/bzhangGo/rmsnorm)  
32. RMSNorm: The Simplified Powerhouse Behind Modern LLMs | Kaggle, accessed on August 26, 2025, [https://www.kaggle.com/discussions/general/556029](https://www.kaggle.com/discussions/general/556029)  
33. Root Mean Square Layer Normalization \- arXiv, accessed on August 26, 2025, [https://arxiv.org/pdf/1910.07467](https://arxiv.org/pdf/1910.07467)  
34. \[1910.07467\] Root Mean Square Layer Normalization \- arXiv, accessed on August 26, 2025, [https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)  
35. Xavier and He Normal (He-et-al) Initialization | by Vishnu Kakaraparthi \- Medium, accessed on August 26, 2025, [https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528](https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528)  
36. Xavier initialization \- GeeksforGeeks, accessed on August 26, 2025, [https://www.geeksforgeeks.org/deep-learning/xavier-initialization/](https://www.geeksforgeeks.org/deep-learning/xavier-initialization/)  
37. Xavier and he\_normal initialization difference \- Stack Overflow, accessed on August 26, 2025, [https://stackoverflow.com/questions/48641192/xavier-and-he-normal-initialization-difference](https://stackoverflow.com/questions/48641192/xavier-and-he-normal-initialization-difference)  
38. Mastering Xavier Initialization: Enhancing Neural Networks for Optimal Training \- LunarTech, accessed on August 26, 2025, [https://www.lunartech.ai/blog/mastering-xavier-initialization-enhancing-neural-networks-for-optimal-training](https://www.lunartech.ai/blog/mastering-xavier-initialization-enhancing-neural-networks-for-optimal-training)  
39. He and Xavier Initialization Functions | by Francesco Franco | The Deep Hub \- Medium, accessed on August 26, 2025, [https://medium.com/thedeephub/he-and-xavier-weight-initialization-functions-acedc5322ce5](https://medium.com/thedeephub/he-and-xavier-weight-initialization-functions-acedc5322ce5)  
40. What is Flash Attention? \- Hopsworks, accessed on August 26, 2025, [https://www.hopsworks.ai/dictionary/flash-attention](https://www.hopsworks.ai/dictionary/flash-attention)  
41. FlashAttention: Fast Transformer training with long sequences \- Adept AI, accessed on August 26, 2025, [https://www.adept.ai/blog/flashier-attention](https://www.adept.ai/blog/flashier-attention)  
42. Flash Attention \- Hugging Face, accessed on August 26, 2025, [https://huggingface.co/docs/text-generation-inference/conceptual/flash\_attention](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention)  
43. The Evolution of Flash Attention: Revolutionizing Transformer Efficiency | by Saiii | Medium, accessed on August 26, 2025, [https://medium.com/@sailakkshmiallada/the-evolution-of-flash-attention-revolutionizing-transformer-efficiency-8a039918d507](https://medium.com/@sailakkshmiallada/the-evolution-of-flash-attention-revolutionizing-transformer-efficiency-8a039918d507)  
44. Flash attention(Fast and Memory-Efficient Exact Attention with IO-Awareness): A deep dive, accessed on August 26, 2025, [https://towardsdatascience.com/flash-attention-fast-and-memory-efficient-exact-attention-with-io-awareness-a-deep-dive-724af489997b/](https://towardsdatascience.com/flash-attention-fast-and-memory-efficient-exact-attention-with-io-awareness-a-deep-dive-724af489997b/)  
45. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning \- arXiv, accessed on August 26, 2025, [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)  
46. What is Flash Attention? | Modal Blog, accessed on August 26, 2025, [https://modal.com/blog/flash-attention-article](https://modal.com/blog/flash-attention-article)  
47. AdamW \- Hugging Face, accessed on August 26, 2025, [https://huggingface.co/docs/bitsandbytes/reference/optim/adamw](https://huggingface.co/docs/bitsandbytes/reference/optim/adamw)  
48. Does the default weight\_decay of 0.0 in transformers.AdamW make sense? \- Models, accessed on August 26, 2025, [https://discuss.huggingface.co/t/does-the-default-weight-decay-of-0-0-in-transformers-adamw-make-sense/1180](https://discuss.huggingface.co/t/does-the-default-weight-decay-of-0-0-in-transformers-adamw-make-sense/1180)  
49. \[R\] Why do we need weight decay in modern deep learning? : r/MachineLearning \- Reddit, accessed on August 26, 2025, [https://www.reddit.com/r/MachineLearning/comments/173vy9t/r\_why\_do\_we\_need\_weight\_decay\_in\_modern\_deep/](https://www.reddit.com/r/MachineLearning/comments/173vy9t/r_why_do_we_need_weight_decay_in_modern_deep/)  
50. apxml.com, accessed on August 26, 2025, [https://apxml.com/posts/transformer-model-regularization-techniques\#:\~:text=Dropout%20is%20a%20frequently%20used,zero%20during%20each%20training%20update.](https://apxml.com/posts/transformer-model-regularization-techniques#:~:text=Dropout%20is%20a%20frequently%20used,zero%20during%20each%20training%20update.)  
51. 5.6. Dropout — Dive into Deep Learning 1.0.3 documentation, accessed on August 26, 2025, [http://d2l.ai/chapter\_multilayer-perceptrons/dropout.html](http://d2l.ai/chapter_multilayer-perceptrons/dropout.html)  
52. Top 6 Regularization Techniques for Transformer Models \- ApX Machine Learning, accessed on August 26, 2025, [https://apxml.com/posts/transformer-model-regularization-techniques](https://apxml.com/posts/transformer-model-regularization-techniques)  
53. Residual Dropout: A Simple Approach to Improve Transformer's Data, accessed on August 26, 2025, [https://aclanthology.org/2024.sigul-1.35.pdf](https://aclanthology.org/2024.sigul-1.35.pdf)  
54. The Role of Dropout in Neural Networks | by Amit Yadav | Biased-Algorithms \- Medium, accessed on August 26, 2025, [https://medium.com/biased-algorithms/the-role-of-dropout-in-neural-networks-fffbaa77eee7](https://medium.com/biased-algorithms/the-role-of-dropout-in-neural-networks-fffbaa77eee7)  
55. Epochs, Batch Size, Iterations \- How are They Important to Training AI and Deep Learning Models \- SabrePC, accessed on August 26, 2025, [https://www.sabrepc.com/blog/Deep-Learning-and-AI/Epochs-Batch-Size-Iterations](https://www.sabrepc.com/blog/Deep-Learning-and-AI/Epochs-Batch-Size-Iterations)  
56. Batch Size in Neural Network \- GeeksforGeeks, accessed on August 26, 2025, [https://www.geeksforgeeks.org/deep-learning/batch-size-in-neural-network/](https://www.geeksforgeeks.org/deep-learning/batch-size-in-neural-network/)  
57. Disadvantages of using very large batch size \- Beginner (2018) \- fast.ai Course Forums, accessed on August 26, 2025, [https://forums.fast.ai/t/disadvantages-of-using-very-large-batch-size/29177](https://forums.fast.ai/t/disadvantages-of-using-very-large-batch-size/29177)  
58. How does Batch Size impact your model learning | by Devansh | Geek Culture \- Medium, accessed on August 26, 2025, [https://medium.com/geekculture/how-does-batch-size-impact-your-model-learning-2dd34d9fb1fa](https://medium.com/geekculture/how-does-batch-size-impact-your-model-learning-2dd34d9fb1fa)  
59. What is the trade-off between batch size and number of iterations to train a neural network?, accessed on August 26, 2025, [https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu](https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu)  
60. Why mini batch size is better than one single "batch" with all training data?, accessed on August 26, 2025, [https://datascience.stackexchange.com/questions/16807/why-mini-batch-size-is-better-than-one-single-batch-with-all-training-data](https://datascience.stackexchange.com/questions/16807/why-mini-batch-size-is-better-than-one-single-batch-with-all-training-data)  
61. How do batch sizes affect the performance of transformer-based language models?, accessed on August 26, 2025, [https://massedcompute.com/faq-answers/?question=How%20do%20batch%20sizes%20affect%20the%20performance%20of%20transformer-based%20language%20models?](https://massedcompute.com/faq-answers/?question=How+do+batch+sizes+affect+the+performance+of+transformer-based+language+models?)  
62. The Impact of Depth and Width on Transformer Language Model Generalization, accessed on August 26, 2025, [https://openreview.net/forum?id=WIGsqpZpFT¬eId=NwgqOaTVEQ](https://openreview.net/forum?id=WIGsqpZpFT&noteId=NwgqOaTVEQ)  
63. What is the effect of increasing the number of layers in a transformer model on its performance and size? \- Massed Compute, accessed on August 26, 2025, [https://massedcompute.com/faq-answers/?question=What%20is%20the%20effect%20of%20increasing%20the%20number%20of%20layers%20in%20a%20transformer%20model%20on%20its%20performance%20and%20size?](https://massedcompute.com/faq-answers/?question=What+is+the+effect+of+increasing+the+number+of+layers+in+a+transformer+model+on+its+performance+and+size?)  
64. Unveiling the Transformer: Impact of Layers and Attention Heads in Audio Classification | by Christopher Ibe | Medium, accessed on August 26, 2025, [https://medium.com/@ccibeekeoc42/unveiling-the-transformer-impact-of-layers-and-attention-heads-in-audio-classification-58747d52b794](https://medium.com/@ccibeekeoc42/unveiling-the-transformer-impact-of-layers-and-attention-heads-in-audio-classification-58747d52b794)  
65. Transformer (deep learning architecture) \- Wikipedia, accessed on August 26, 2025, [https://en.wikipedia.org/wiki/Transformer\_(deep\_learning\_architecture)](https://en.wikipedia.org/wiki/Transformer_\(deep_learning_architecture\))  
66. Are Sixteen Heads Really Better than One? \- NIPS, accessed on August 26, 2025, [http://papers.neurips.cc/paper/9551-are-sixteen-heads-really-better-than-one.pdf](http://papers.neurips.cc/paper/9551-are-sixteen-heads-really-better-than-one.pdf)  
67. The Story of Heads \- Lena Voita, accessed on August 26, 2025, [https://lena-voita.github.io/posts/acl19\_heads.html](https://lena-voita.github.io/posts/acl19_heads.html)  
68. \[D\] Batch size vs learning rate : r/MachineLearning \- Reddit, accessed on August 26, 2025, [https://www.reddit.com/r/MachineLearning/comments/1fqqfos/d\_batch\_size\_vs\_learning\_rate/](https://www.reddit.com/r/MachineLearning/comments/1fqqfos/d_batch_size_vs_learning_rate/)