import inspect
import torch
import torch.nn as nn
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.embedding import TokenAndPositionEmbedding
from jimmylabs.model.block import TransformerBlock
import math

class GPT(nn.Module):
    """
    The full JimmyLabs GPT model.
    Follows SPEC.md §2 and docs/10_GPT_ARCHITECTURE.md.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = TokenAndPositionEmbedding(config),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (ADR-0003)
        if config.weight_tying:
            self.lm_head.weight = self.transformer.wte.token_embedding.weight
            
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, use_cache: bool = False, past_key_values: list = None):
        """
        Args:
            idx: (B, T) tensor of integer token indices
            targets: (B, T) tensor of integer token indices (optional)
            use_cache: whether to return KV cache state
            past_key_values: list of layer KV states (optional)
        Returns:
            logits: (B, T, V) tensor
            loss: cross entropy loss (scalar) if targets provided, else None
            past_key_values: list of KV states if use_cache=True
        """
        # Embeddings
        pos_offset = past_key_values[0][0].size(-2) if past_key_values is not None else 0
        x = self.transformer.wte(idx, pos_offset=pos_offset)
        
        presents = [] if use_cache else None
        
        # Transformer blocks
        for i, block in enumerate(self.transformer.h):
            past = past_key_values[i] if past_key_values is not None else None
            
            if use_cache:
                x, present = block(x, use_cache=True, past_key_value=past)
                presents.append(present)
            else:
                x = block(x, use_cache=False, past_key_value=past)
            
        # Final LayerNorm
        x = self.transformer.ln_f(x)
        
        # LM Head
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Flatten for cross_entropy: (B*T, V) and (B*T)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        if use_cache:
            return logits, loss, presents
        return logits, loss
        
    def configure_optimizers(self, weight_decay: float, learning_rate: float, device_type: str) -> torch.optim.Optimizer:
        """
        Configure AdamW optimizer. Applies weight decay to 2D parameters (matmuls)
        but NOT to 1D parameters (LayerNorms, biases) to prevent degrading normalization.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                    
        # subtle: 'lm_head.weight' is tied to 'transformer.wte.weight'
        # so it appeared in no_decay. We should make sure we don't put it in decay if weight_tying is on
        # actually, because of weight tying, the param is exactly the embedding param.
        # we can just use the union of sets, but let's build dict of param name -> param
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # because of weight tying, 'lm_head.weight' might be in decay but not in param_dict
        # because named_parameters() removes duplicates by default.
        if 'lm_head.weight' in decay and 'lm_head.weight' not in param_dict:
            decay.remove('lm_head.weight')
            
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        # validate that we considered every parameter
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        
        # create the pytorch optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        # Use fused AdamW on MPS/CUDA if available for speed.
        # `fused` is a constructor keyword, not an attribute of the class, so
        # hasattr(torch.optim.AdamW, 'fused') is False on every torch build --
        # the previous check disabled this path everywhere, silently.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = device_type in ('cuda', 'mps') and fused_available
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)
        
        return optimizer

