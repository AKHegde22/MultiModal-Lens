"""Weight processing utilities for rigorous mechanistic interpretability.

These utilities transform model weights to make residual stream decomposition
exact, following TransformerLens methodology.
"""

from __future__ import annotations
import torch
import torch.nn as nn

def fold_layer_norms(model: nn.Module) -> None:
    """Absorb LayerNorm learnable parameters into adjacent linear weights.
    
    For LayerNorm(x) = gamma * (x - mu) / sigma + beta followed by linear W:
      W_folded = diag(gamma) @ W  (i.e. W_folded[i,:] = gamma[i] * W[i,:])
      b_folded = beta @ W + b_original
    
    After folding, the LayerNorm gamma/beta are set to 1/0 respectively,
    making it a pure mean-center + variance-scale operation.
    This makes the residual stream decomposition exact.
    
    Modifies model weights in-place.
    """
    for name, module in model.named_modules():
        # Heuristic for generic transformer blocks
        if hasattr(module, 'input_layernorm') and hasattr(module, 'self_attn'):
            ln = getattr(module, 'input_layernorm')
            attn = getattr(module, 'self_attn')
            
            linears = [m for m in attn.modules() if isinstance(m, nn.Linear)]
            if not linears and hasattr(attn, 'q_proj'):
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'c_attn']:
                    if hasattr(attn, proj_name):
                        linears.append(getattr(attn, proj_name))
            
            if hasattr(ln, 'weight') and ln.weight is not None:
                gamma = ln.weight.data
                beta = ln.bias.data if hasattr(ln, 'bias') and ln.bias is not None else torch.zeros_like(gamma)
                
                for lin in linears:
                    with torch.no_grad():
                        lin.weight.data = lin.weight.data * gamma.view(1, -1)
                        if lin.bias is not None:
                            lin.bias.data = lin.bias.data + torch.matmul(lin.weight.data, beta)
                        else:
                            lin.bias = nn.Parameter(torch.matmul(lin.weight.data, beta))
                
                with torch.no_grad():
                    ln.weight.data.fill_(1.0)
                    if hasattr(ln, 'bias') and ln.bias is not None:
                        ln.bias.data.fill_(0.0)
                        
        if hasattr(module, 'post_attention_layernorm') and hasattr(module, 'mlp'):
            ln = getattr(module, 'post_attention_layernorm')
            mlp = getattr(module, 'mlp')
            
            linears = []
            for proj_name in ['gate_proj', 'up_proj', 'c_fc', 'fc_in']:
                if hasattr(mlp, proj_name):
                    linears.append(getattr(mlp, proj_name))
            
            if not linears:
                 linears = [m for n, m in mlp.named_modules() if isinstance(m, nn.Linear) and ('out' not in n and 'down' not in n and 'c_proj' not in n)]

            if hasattr(ln, 'weight') and ln.weight is not None:
                gamma = ln.weight.data
                beta = ln.bias.data if hasattr(ln, 'bias') and ln.bias is not None else torch.zeros_like(gamma)
                
                for lin in linears:
                    with torch.no_grad():
                        lin.weight.data = lin.weight.data * gamma.view(1, -1)
                        if lin.bias is not None:
                            lin.bias.data = lin.bias.data + torch.matmul(lin.weight.data, beta)
                        else:
                            lin.bias = nn.Parameter(torch.matmul(lin.weight.data, beta))
                
                with torch.no_grad():
                    ln.weight.data.fill_(1.0)
                    if hasattr(ln, 'bias') and ln.bias is not None:
                        ln.bias.data.fill_(0.0)

def center_writing_weights(model: nn.Module) -> None:
    """Mean-center columns of residual-stream-writing weight matrices.
    
    Since downstream LayerNorm removes the mean, any constant offset
    in residual writes is a free parameter that doesn't affect model
    behavior but skews DLA results.
    
    Specifically centers:
    - W_O (attention output projection): each head's output writing
    - W_out (MLP output projection): MLP's residual contribution
    - Embedding matrix: initial residual stream
    
    W_centered = W - mean(W, dim=0, keepdim=True)
    
    Modifies model weights in-place.
    """
    write_proj_names = ['o_proj', 'out_proj', 'down_proj', 'c_proj']
    
    for name, module in model.named_modules():
        if any(name.endswith(proj) for proj in write_proj_names) and isinstance(module, nn.Linear):
            with torch.no_grad():
                mean_w = module.weight.data.mean(dim=0, keepdim=True)
                module.weight.data = module.weight.data - mean_w
                if module.bias is not None:
                    module.bias.data = module.bias.data - module.bias.data.mean()
                    
        if isinstance(module, nn.Embedding) and ('embed' in name.lower() or 'wte' in name.lower()):
            with torch.no_grad():
                mean_w = module.weight.data.mean(dim=0, keepdim=True)
                module.weight.data = module.weight.data - mean_w

def center_unembed(model: nn.Module) -> None:
    """Mean-center rows of the unembedding matrix W_U.
    
    Removes the 'bias' direction that equally boosts all logits.
    This doesn't change relative logit differences (and thus predictions)
    but makes DLA attribution cleaner.
    
    W_U_centered = W_U - mean(W_U, dim=0, keepdim=True)
    
    Modifies model weights in-place.
    """
    unembed_names = ['lm_head', 'embed_out', 'output']
    for name, module in model.named_modules():
        if any(name.endswith(unp) for unp in unembed_names) or (hasattr(model, 'get_output_embeddings') and module is getattr(model, 'get_output_embeddings', lambda: None)()):
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    mean_w = module.weight.data.mean(dim=0, keepdim=True)
                    module.weight.data = module.weight.data - mean_w
                    if module.bias is not None:
                        module.bias.data = module.bias.data - module.bias.data.mean()
