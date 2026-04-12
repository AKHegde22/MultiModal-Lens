# MultimodalLens: A Lightweight Interactive Debugger for Vision-Language Models

## Abstract

Vision-language models (VLMs) are increasingly used in high-stakes settings, yet practical debugging tools remain fragmented across architectures. We present **MultimodalLens**, a lightweight web-based debugger that unifies attention visualization, token-image alignment analysis, and faithfulness diagnostics for Hugging Face multimodal models. The system introduces an adapterized backend supporting dual-encoder, Q-Former, and interleaved decoder VLM families, and a Gradio interface with synchronized exploration, comparison, and batch evaluation workflows. Beyond visualization, MultimodalLens integrates perturbation-based diagnostics (deletion/insertion, counterfactual masking, and attention-gradient agreement) to mitigate over-interpretation of attention heatmaps. We outline the implementation, discuss methodological tradeoffs, and propose a reproducible evaluation protocol for grounding and hallucination analysis.

## 1. Introduction

Modern multimodal transformers expose rich internals (attentions, hidden states, and embeddings), but interpretability tooling remains architecture-specific and difficult to operationalize. Existing tools either focus on language-only attention inspection or target narrow VLM setups without a unified debugging workflow.

MultimodalLens addresses this gap by providing:

1. A common analysis interface across major VLM architecture families.
2. Side-by-side visual evidence: attention, alignment, and similarity.
3. Built-in faithfulness probes to validate explanation quality.
4. A lightweight deployment path using Gradio and standard Hugging Face APIs.

## 2. Related Work

### 2.1 Transformer attention visualization

BertViz popularized interactive head/layer inspection for transformer attention and established practical workflows for model introspection. However, subsequent studies highlighted that attention alone is not a faithful explanation signal and can diverge from causal importance.

### 2.2 Attention attribution and faithfulness

Transformer attribution methods (e.g., attention rollout, relevance propagation, gradient-based methods) improve interpretability but still require calibration with perturbation tests. Deletion/insertion protocols and counterfactual interventions provide stronger behavioral evidence.

### 2.3 VLM interpretability tools

Recent VLM-oriented tools demonstrate architecture-specific visualization (e.g., cross-modal attention exploration), but general-purpose, HF-native, multi-architecture debuggers with integrated faithfulness checks are still limited.

## 3. Problem Formulation

Given an input pair `(x_img, x_txt)` and model `f`, we seek a diagnostic tuple:

`D = (A, S, g, F)`

where:

- `A`: attention-derived maps
- `S`: token-image similarity matrix
- `g`: scalar global cross-modal score
- `F`: faithfulness metrics under targeted perturbations

The objective is to support developer reasoning about errors, hallucinations, and grounding failures with multiple convergent signals.

## 4. Method

### 4.1 Adapterized architecture

We define a `ModelAdapter` API with family-specific implementations:

- **Dual encoder (CLIP-like)**: independent text/vision towers and direct embedding similarity.
- **Q-Former (BLIP-2-like)**: vision encoder + query transformer + language model stack.
- **Interleaved decoder (LLaVA-like)**: image-conditioned token streams in autoregressive decoder.

All adapters return a normalized `AnalysisResult` object.

### 4.2 Attention channel

For available attention tensors, we compute residual-augmented rollout and extract query-to-image-token relevance maps, then reshape to a patch grid for image overlay.

### 4.3 Alignment channel

We compute token-patch cosine similarity in shared latent space:

`S_ij = cos(t_i, v_j)`

where `t_i` and `v_j` are text token and visual patch vectors respectively.

### 4.4 Faithfulness channel

We evaluate explanation plausibility via:

1. Deletion curve: score change under progressive masking of top-ranked patches.
2. Insertion curve: score recovery under complementary masking.
3. Counterfactual drop: single-step top-k masking effect.
4. Attention-gradient rank agreement: Spearman correlation.

## 5. System Implementation

### 5.1 Frontend

A Gradio app provides three tabs:

- **Explore**: single example deep inspection.
- **Compare**: prompt-level differential analysis.
- **Eval**: dataset-level batch execution from JSONL.

### 5.2 Backend

Core modules include:

- `adapters/*`: family-specific extraction
- `analysis/*`: attribution/alignment/faithfulness
- `core/pipeline.py`: adapter caching and orchestration
- `eval/runner.py`: batch metrics export

## 6. Experimental Protocol (Initial)

### 6.1 Model set

- `openai/clip-vit-base-patch32`
- `Salesforce/blip2-opt-2.7b`
- `llava-hf/llava-1.5-7b-hf`

### 6.2 Proposed evaluation datasets

- Grounding-focused subsets from Flickr30k Entities and RefCOCO variants
- Hallucination-focused prompts following POPE-style templates
- Caption compatibility metrics (e.g., CLIPScore-like analyses)

### 6.3 Quantitative outputs

Per sample:

- global score
- counterfactual drop
- deletion/insertion AUC proxies
- attention-gradient agreement

Aggregate reports should include model-wise mean/variance and failure taxonomy.

## 7. Discussion

MultimodalLens emphasizes practical debugging over single-method explanation claims. By combining representation and perturbation evidence, the system is more robust against misleading heatmaps. Limitations include variability in exposed internals across model checkpoints and imperfect score proxies for generative settings.

## 8. Conclusion

We present a practical, extensible, and architecture-aware VLM debugger with integrated faithfulness checks. The tool is suitable for rapid model development cycles and can be extended into a standardized evaluation suite for multimodal reliability.

## References (Indicative)

1. Vig, J. (2019). A Multiscale Visualization of Attention in the Transformer Model.
2. Jain, S., and Wallace, B. (2019). Attention is not Explanation.
3. Abnar, S., and Zuidema, W. (2020). Quantifying Attention Flow in Transformers.
4. Chefer, H., et al. (2021). Transformer Interpretability Beyond Attention Visualization.
5. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP).
6. Li, J., et al. (2022). BLIP: Bootstrapping Language-Image Pre-training.
7. Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.
8. Liu, H., et al. (2023). Visual Instruction Tuning (LLaVA).
9. Wang, J., et al. (2023). POPE: Polling-based Object Probing Evaluation for Hallucination.
