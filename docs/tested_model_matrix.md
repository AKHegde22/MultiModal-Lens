# Tested Model Matrix

This matrix tracks validated behavior per model checkpoint.

Legend:
- PASS: validated by manual run and/or regression workflow
- PARTIAL: core analysis works, some mechanistic probes may vary by backend
- EXPERIMENTAL: routed via alias/auto inference; not fully smoke-tested yet

| Family label | Example checkpoint | Explore/Compare/Eval | Mechanistic suite | Status | Notes |
|---|---|---|---|---|---|
| clip | openai/clip-vit-base-patch32 | Yes | Yes | PASS | CLIP logit lens + grounding fixed with eager attention path. |
| blip2 | Salesforce/blip2-opt-2.7b | Yes | Yes | PARTIAL | Depends on checkpoint memory footprint and qformer output availability. |
| llava | llava-hf/llava-1.5-7b-hf | Yes | Yes | PARTIAL | Core behavior validated; large models may need device tuning. |
| qwen2_vl | Qwen/Qwen2-VL-2B-Instruct | Expected | Expected | EXPERIMENTAL | Routed through llava-style adapter path. |
| idefics2 | HuggingFaceM4/idefics2-8b | Expected | Expected | EXPERIMENTAL | Routed through llava-style adapter path. |
| paligemma | google/paligemma-3b-mix-224 | Expected | Expected | EXPERIMENTAL | Routed through llava-style adapter path. |
| mllama | meta-llama/Llama-3.2-11B-Vision-Instruct | Expected | Expected | EXPERIMENTAL | Requires large GPU memory. |
| internvl | OpenGVLab/InternVL2_5-2B | Expected | Expected | EXPERIMENTAL | Routed through llava-style adapter path. |

## How To Update

1. Run preflight:
   - `python scripts/run_preflight.py --family auto --model <checkpoint>`
2. Run one Explore pass and (optional) mechanistic suite in UI or CLI.
3. Update this table with observed status and notes.
