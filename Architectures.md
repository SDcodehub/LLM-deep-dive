This lecture, "Everything You Didn't Want To Know About LM Architecture And Training," dives into the specific details of **Large Language Model (LLM) architectures** and their **hyperparameters**, drawing insights from the extensive empirical work in the field. The goal is to understand what design choices are critical for building effective transformers.

---

## Transformer Architecture Variants

The lecture begins with a recap of the **original transformer** and then introduces a **modern consensus variant** that students implement. It then takes a data-driven approach, analyzing recent LLM releases to identify convergent evolution in architectural choices.

### Layer Normalization (LayerNorm) Placement

* **Original Transformer (Post-Norm):** LayerNorm applied *after* each sub-component (multi-head attention, MLP) and residual addition.
* **Modern LMs (Pre-Norm):** LayerNorm moved *before* the multi-head attention and MLP blocks, within the residual stream. This is a near-universal choice due to **improved training stability** and **better gradient propagation**.
    * **Reasoning:** Post-norm was found to be less stable, often requiring careful learning rate warm-up. Pre-norm generally leads to more stable training, fewer loss spikes, and more consistent gradient norms.
* **Double Norm (Recent Innovation):** Some recent models like **Grok** and **Gemma 2** incorporate LayerNorms both in front *and* after the blocks. **OLMo 2** uses LayerNorm only after the feed-forward and multi-head attention. This indicates ongoing exploration in this area.

### Normalization Type: LayerNorm vs. RMS Norm

* **LayerNorm:** Normalizes by subtracting the empirical mean and dividing by the standard deviation, then scales by a learnable $\gamma$ and shifts by a $\beta$.
* **RMS Norm:** A simplified version of LayerNorm that **drops the mean adjustment and the bias term ($\beta$)**.
    * **Consensus:** Most modern LLMs (**LLaMA family, PaLM, Chinchilla, T5s**) have switched to RMS Norm.
    * **Reasoning:**
        * **Performance:** RMS Norm performs comparably to LayerNorm.
        * **Efficiency:** Fewer operations (no mean subtraction) and fewer parameters (no bias term $\beta$) lead to faster execution.
        * **Memory Movement:** While tensor contractions (matrix multiplies) account for ~99.8% of FLOPs in a transformer, normalization operations (like LayerNorm/RMS Norm) can consume ~25% of runtime due to **memory movement overhead**. Reducing operations and parameters directly reduces this overhead.
* **Bias Terms:** Most modern transformers **omit bias terms** in their linear layers.
    * **Reasoning:** They perform just as well without them. More importantly, removing bias terms often **stabilizes the training** of large neural networks.

### Activation Functions and Gating Mechanisms

* **ReLU (Rectified Linear Unit):** $\text{max}(0, x)$. Used in original transformer and some older models.
* **GeLU (Gaussian Error Linear Unit):** Multiplies linear input with CDF of a Gaussian. Provides a smoother, more differentiable activation. Used by **GPT-1, 2, 3, GPT-J**.
* **Gated Linear Units (GLUs):** This category includes **SwiGLU, GeGLU, ReGLU**.
    * **Mechanism:** Gating involves multiplying the hidden part of the MLP with an element-wise linear term. For example, a ReGLU uses a ReLU for the non-linearity and gates it. A GeGLU uses a GeLU for the non-linearity with gating. A SwiGLU uses a Swish function ($x \cdot \text{sigmoid}(x)$) as the non-linearity with gating.
    * **Consensus:** **SwiGLU** and **GeGLU** variants are **widespread and dominant** in modern models (**LLaMA family, PaLM, OlMo, T5v1.1, Gemma 2, Gemma 3**).
    * **Reasoning:** Empirical evidence consistently shows **performance gains** from GLU variants.
    * **Parameter Sizing:** Gated models typically scale down the hidden size by a factor of **2/3** to ensure the total number of parameters remains similar to non-gated counterparts.

### Serial vs. Parallel Layers

* **Serial Layers:** In each transformer block, attention computation is performed first, followed by the MLP. This is the **most common approach**.
* **Parallel Layers:** Attention and MLP computations are performed **simultaneously** and then added to the residual stream.
    * **Pioneers:** **GPT-J** and **PaLM** adopted this.
    * **Reasoning:** Can offer **system efficiencies** by fusing matrix multiplies and potentially better GPU utilization, especially for very large-scale training.
    * **Prevalence:** Less common in recent models, with exceptions like **Cohere Command A, Command R+, and Falcon 2 11B**.

---

## Position Embeddings

* **Original Transformer:** Used **sine and cosine position embeddings**.
* **Absolute Embeddings:** **GPTs, OPT** added learned position vectors.
* **Relative Embeddings:** **T5, Gopher** used relative embeddings that modify attention computations.
* **RoPE (Rotary Position Embeddings):**
    * **Mechanism:** Exploits the property that inner products are invariant to arbitrary rotations. It rotates pairs of dimensions in the query and key vectors based on their relative positions.
    * **Placement:** RoPE operates directly at the **attention layer** (on queries and keys) rather than adding position embeddings at the input layer. This enforces relative positioning only.
    * **Consensus:** RoPE has seen **convergent evolution** and is now used by **almost all recent LLMs** (including the **LLaMA family, Mistral, Gemma, Qwen, DeepSeek, Yi**).
    * **Reasoning:** Empirically effective and supports various algorithms for **extrapolating context length**, which is crucial for modern LLMs.

---

## Hyperparameter Selection

The lecture emphasizes that despite the vast number of hyperparameters, clear rules of thumb have emerged.

### Feed-Forward (FFN) Dimension Ratio

* **General Rule:** For ReLU-style MLPs, `d_ff = 4 * d_model`.
* **GLU Variants:** For GLU variants, the ratio is approximately `d_ff = 8/3 * d_model` (or `~2.66 * d_model`) to parameter-match with the 4x multiplier of non-gated units.
    * **Evidence:** Kaplan et al.'s scaling law paper shows a wide basin of optimal performance for `d_ff / d_model` ratios between 1 and 10, with 4 (or 2.66 for GLU) being a reasonable choice near the optimum.
* **Exceptions:** **T5-11B** famously used a `d_ff` 64 times larger than `d_model`. While T5 was a fine model, its successor **T5v1.1** reverted to a more standard 2.5x multiplier, suggesting the extreme ratio wasn't optimal. **Gemma 2** uses a factor of 8.

### Model Dimension to Head Dimension Ratio

* **Canonical Choice:** `d_model / (d_head * num_heads) = 1`. This means the `d_model` (hidden dimension) is split equally among the heads.
* **Prevalence:** **GPT-3, T5, LaMDA, PaLM, LLaMA2** all follow this ratio.
* **Exceptions:** T5 is a notable exception with a ratio of 16.
* **Practicality:** While some research suggests benefits for differing ratios (e.g., more heads with fewer dimensions per head), in practice, a 1:1 ratio has not shown significant low-rank bottlenecks.

### Aspect Ratio (Width vs. Depth)

* **Definition:** `d_model / n_layers` (hidden dimensions per layer).
* **Sweet Spot:** Around **128 hidden dimensions per layer** has been a general consensus for many models (**GPT-3, LLaMA variants**).
* **Kaplan et al. Evidence:** Performance optima for aspect ratio remain relatively consistent across different model scales, generally around 100-200.
* **System Considerations:** Aspect ratio impacts parallelism strategies (pipeline parallel for deep models, tensor parallel for wide models), which in turn depend on networking constraints.
* **Downstream Performance:** Some research suggests that for the same FLOP count, deeper models might perform better on downstream tasks, even if raw loss is similar.

### Vocabulary Size

* **Trend:** Vocabulary sizes have been **trending upwards**.
* **Early/Monolingual Models:** 30,000-50,000 tokens (**GPTs, early LLaMAs**).
* **Modern Multilingual/Production Systems:** **100,000-250,000 tokens**.
    * **Examples:** **Cohere Command A** (emphasizes multilingual capabilities), **GPT-4** tokenizer (`~100k` tokens).
* **Reasoning:** Larger vocabularies better handle diverse languages, emojis, and other input types, especially as LLMs are deployed for broader use cases. They can also represent minority languages more efficiently (fewer tokens).

### Regularization: Dropout and Weight Decay

* **Pre-training Context:** In pre-training, models typically perform one epoch over vast datasets, making overfitting less of a concern. This suggests regularization might not be necessary.
* **Dropout:** Has generally **gone out of fashion** for pre-training.
* **Weight Decay:** Still widely used, which is counter-intuitive for pre-training.
    * **Reasoning:** Weight decay does **not primarily control overfitting** in pre-training (it doesn't significantly change the train-to-validation loss gap). Instead, it **interacts with learning rate schedules** to achieve **better training losses** (and consequently, better validation losses). Models with high weight decay can start slow but rapidly optimize as the learning rate decreases, suggesting an implicit acceleration effect towards the end of training.

---

## Stability Interventions

Training large models for extended periods surfaces stability issues, particularly gradient explosions. Recent innovations focus on mitigating these.

### Problem Area: Softmax

* Softmax operations are prone to numerical instability due to exponentiation and potential division by zero.
* **Locations in Transformer:** Output softmax (final layer), Self-attention softmax.

### Z-Loss

* **Mechanism:** Adds an auxiliary loss term to encourage the softmax normalizer ($Z$) to be close to 1 (or $\log Z$ close to 0). This makes the softmax numerically more stable.
* **Pioneers:** **PaLM** first used this (`10^{-4} \log^2 Z`).
* **Adoption:** **Baichuan 2, DCLM, OLMo 2**, and others have adopted z-loss.

### QK Norm (Query-Key Normalization)

* **Mechanism:** Applies a **LayerNorm** to the queries (Q) and keys (K) *before* their inner product is computed for the softmax in the attention mechanism. This bounds the inputs to the softmax, controlling its behavior.
* **Origin:** Innovation from vision and multimodal model communities (**Dehgani 2023, Chameleon, Idefcs**).
* **Adoption:** **Gemma 2, DCLM, OLMo 2** use this.
* **Effectiveness:** LayerNorms prove surprisingly effective in improving training stability without significantly affecting performance.

### Soft-Capping Logits

* **Mechanism:** Applies a `tanh` function to the inner product (logits) of the QK attention mechanism, effectively clipping their maximum value. This prevents extreme values from going into the softmax.
* **Adoption:** **Gemma 2, OLMo 2** use this.
* **Effectiveness:** NVIDIA's research suggests soft-capping can sometimes worsen perplexity, while QK norm tends to improve it by allowing more aggressive learning rates.

---

## Attention Head Variations

These variations are less about core architecture and more about **inference time efficiency** and **context length handling**.

### Multi-Query Attention (MQA) and Grouped Query Attention (GQA)

* **Problem:** At inference time, **KV cache** (storing past keys and values for incremental attention computation) leads to high **memory access costs** and poor arithmetic intensity, especially with large sequence lengths and smaller batches. This is because the key and value matrices grow with sequence length.
* **MQA:** Instead of having a separate key and value head for each query head, MQA uses **multiple query heads but only one set of key and value heads**.
    * **Benefit:** Drastically reduces memory movement for keys and values, significantly improving arithmetic intensity and inference throughput, especially for longer sequence lengths.
* **GQA:** A generalization of MQA. Instead of a single KV head, it uses a **small group of KV heads** (e.g., 8 KV heads for 64 query heads).
    * **Benefit:** Trades off between the expressiveness of multi-head attention and the efficiency of MQA, often finding a good balance.

### Sparse Attention Patterns

* **Motivation:** Standard self-attention (quadratic complexity in sequence length) becomes computationally prohibitive for very long contexts.
* **Historical Approaches (2019):**
    * **Local Window Attention:** Each token only attends to a small window around itself.
    * **Diagonal Attention:** Specific attention patterns designed to propagate information across the sequence.
* **Sliding Window Attention:** At each layer, attention is limited to a small, fixed-size window around the current position. The effective receptive field grows with depth.
* **Modern Instantiation (LLaMA 4, Gemma, Cohere Command A):**
    * Combine full self-attention (without position embeddings like RoPE) at a lower frequency (e.g., every 4 blocks) with sliding window attention (with RoPE) in the intermediate blocks.
    * **Benefit:** The full self-attention blocks handle long-range dependencies without positional encodings (allowing aggressive length extrapolation), while sliding window attention with RoPE handles local context efficiently. This balances computational cost with the ability to manage very long contexts (e.g., 10 million tokens).

---
