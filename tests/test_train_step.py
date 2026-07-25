import pytest
import torch
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT
from jimmylabs.utils.seed import seed_everything

def test_train_step_decreases_loss():
    """
    TRAIN-STEP-DECREASES:
    N steps on one fixed batch strictly lower the loss.
    Verifies that configure_optimizers (weight decay rules) and
    a single step of optimization successfully updates weights in the right direction.
    """
    seed_everything(42)
    
    config = GPTConfig(
        vocab_size=65,
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=16,
        dropout=0.0, # Disable dropout for deterministic gradient steps
        weight_tying=True
    )
    model = GPT(config)
    model.train()
    
    # Use configure_optimizers to test weight decay separation logic
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3, device_type='cpu')
    
    B, T = 2, 16
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    
    # 1st step
    optimizer.zero_grad(set_to_none=True)
    _, initial_loss = model(idx, targets)
    initial_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Run a few more steps
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(idx, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    # Final step
    optimizer.zero_grad(set_to_none=True)
    _, final_loss = model(idx, targets)
    
    assert final_loss.item() < initial_loss.item(), f"Loss did not decrease: {initial_loss.item()} -> {final_loss.item()}"


def test_fused_adamw_detection_matches_torch_support():
    """
    FUSED-DETECTION:
    configure_optimizers must actually enable fused AdamW where torch supports it.

    Regression guard: the check used to be
        hasattr(torch.optim.AdamW, 'fused')
    but `fused` is a constructor keyword, not a class attribute, so that is False
    on every torch build -- the fused path was silently dead everywhere. Assert
    against torch's real signature instead of hardcoding an expected value, so
    this stays correct on builds that drop or add fused support.
    """
    import inspect

    seed_everything(42)
    model = GPT(GPTConfig(vocab_size=65, n_layer=2, n_head=2, n_embd=32, block_size=16))
    supported = 'fused' in inspect.signature(torch.optim.AdamW).parameters

    accel = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3, device_type='cuda')
    assert bool(accel.param_groups[0].get('fused', False)) is supported

    # CPU must never take the fused path, regardless of torch support.
    cpu = model.configure_optimizers(weight_decay=0.1, learning_rate=1e-3, device_type='cpu')
    assert not cpu.param_groups[0].get('fused', False)
