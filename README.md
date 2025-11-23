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
