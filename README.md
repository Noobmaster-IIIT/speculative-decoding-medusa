# 🐍 Medusa-Llama: Speculative Decoding with QLoRA Self-Distillation

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)
![Llama-2](https://img.shields.io/badge/Llama--2-7B-blue)

## 🚀 Overview

This repository contains a PyTorch implementation of **Medusa**, a speculative decoding framework that accelerates LLM inference by adding extra "Heads" to predict multiple future tokens simultaneously. 

Unlike the original Medusa implementation, this version utilizes **QLoRA (Quantized Low-Rank Adaptation)** and a **Self-Distillation** training objective, allowing you to train Medusa heads on consumer hardware (e.g., single T4 or A100) without loading a separate teacher model into memory.

## ⚡ Key Architecture

### The Medusa Heads
Instead of a separate draft model, we attach $K$ lightweight MLPs to the final hidden state of the frozen Llama backbone.
- Head 1 predicts token $t+1$
- Head 2 predicts token $t+2$
- ...and so on.

### The Self-Distillation Trick
To save VRAM, we do not load a separate Teacher model. Instead, we use the **LoRA-disabled context** as the teacher:
1.  **Teacher Pass:** Forward pass with LoRA adapters *disabled* (Pure Llama-2).
2.  **Student Pass:** Forward pass with LoRA adapters *enabled*.
3.  **Objective:** Minimize KL Divergence between Student and Teacher, plus Cross-Entropy for the Heads.

```mermaid
graph TD
    subgraph "Inference Flow"
    Input[Input Tokens] --> Backbone[Frozen Llama-2 Backbone]
    Backbone --> Hidden[Final Hidden State]
    Hidden --> LM_Head[Original LM Head]
    Hidden --> Medusa1[Medusa Head 1]
    Hidden --> Medusa2[Medusa Head 2]
    Hidden --> Medusa3[Medusa Head 3]
    
    LM_Head --> Out0[Token t]
    Medusa1 --> Out1[Draft Token t+1]
    Medusa2 --> Out2[Draft Token t+2]
    Medusa3 --> Out3[Draft Token t+3]
    end

graph LR
    Root((Current Token))
    Root --> H1_Opt1(Head1: Option A)
    Root --> H1_Opt2(Head1: Option B)
    H1_Opt1 --> H2_Opt1(Head2: Option C)
    H1_Opt1 --> H2_Opt2(Head2: Option D)
    
    style Root fill:#f9f,stroke:#333
    style H1_Opt1 fill:#bbf,stroke:#333
    style H2_Opt1 fill:#bfb,stroke:#333

## 1. The Core Concept: Why "Medusa"?
Standard autoregressive generation is memory-bound. You move 14GB of weights to the GPU chip to generate **one** token.
* **Idea:** If we are moving 14GB anyway, why not ask the GPU to predict the next **5 tokens** at once?
* **The Catch:** Predicting $t+2$ is hard because you don't know $t+1$ yet.
* **Medusa's Solution:** Train a "Head" that learns: *"Given the hidden state at time $t$, what is the most likely token at $t+2$?"*
    * It's a guess. It might be wrong.
    * If it's right, we accept it (Draft).
    * If it's wrong, we reject it (Verify).

---

## 2. Architecture Deep Dive (`medusa_llama_student.py`)

### The Class: `_MedusaSingleHead`
This is a Residual MLP.
* **Input:** Hidden State $h_t$ of shape `[Batch, Seq, D_model]`.
* **Operation:**
    $$z = \text{SiLU}(W_1 \cdot h_t) + h_t$$
    $$\text{logits} = W_2 \cdot z$$
* **Initialization Magic (`_init_from_lm_head`):**
    * $W_1$ is initialized to **Zero**.
    * $W_2$ is initialized to the **Original LM Head weights**.
    * **Why?** At step 0 of training, `SiLU(0) = 0`. So $z = h_t$. The output is exactly the same as the base model. This makes training extremely stable because the head starts as a "perfect copy" of the base model logic.

### The Wrapper: `MedusaLlamaStudent`
This class wraps the Hugging Face `LlamaForCausalLM`.

**The Forward Pass Journey (`forward` method):**
1.  **Input:** `input_ids` shape `[B, S]`.
2.  **Backbone:** `self.backbone(..., output_hidden_states=True)`.
    * It runs through Llama layers.
    * **Crucial Fix:** You grab `hidden_states[-1]`.
    * Shape: `[B, S, 4096]` (for Llama-7B).
3.  **The Fork (Parallel computation):**
    * **Path A (Standard):** `self.backbone.lm_head(hidden_states)` $\to$ `[B, S, V]`. (The prediction for $t+1$).
    * **Path B (Medusa):** `self.mtp_head(hidden_states)`.
        * Inside `MultiTokenHead`, it loops through offsets (e.g., +1, +2, +3).
        * It flattens input to `[B*S, 4096]` for efficiency.
        * Runs 3 separate MLPs.
        * Returns a Dict: `{1: [B,S,V], 2: [B,S,V], 3: [B,S,V]}`.
4.  **Return:** A dictionary containing *everything* needed for loss calculation.

---

## 3. The Self-Distillation Training Trick (`trainer.py`)

This is the most technically interesting part of your code.

**The Problem:** Distillation usually requires a "Teacher" model (frozen) and a "Student" model (trainable). Loading two Llama-7B models requires ~28GB VRAM (in FP16). Too big for Colab/Consumer GPUs.

**Your Solution:** Use **LoRA** as a switch.
* **Teacher:** The Base Model (LoRA Disabled).
* **Student:** The Base Model + LoRA Adapters (LoRA Enabled).

**Code Walkthrough (Inside `MedusaTrainer.train()`):**

```python
# 1. THE STUDENT PASS
# We are in standard training mode. Adapters are ON.
out_student = self.model(input_ids, ...) 

# 2. THE TEACHER PASS
# "with self.model.backbone.disable_adapter():"
# This is a PEFT context manager. It temporarily tells PyTorch 
# to bypass the LoRA layers A and B and only use Frozen W.
with torch.no_grad():
    with self.model.backbone.disable_adapter(): 
        teacher_out = self.model(input_ids, ...)
