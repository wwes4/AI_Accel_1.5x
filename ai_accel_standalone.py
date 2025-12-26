import torch
import torch.nn as nn
import torch.optim as optim

class AIAccelFramework:
    """
    Synthesized framework: Aggressive pruning for high sparsity/speed + stability fixes.
    """
    def __init__(self, min_tension=1e-3, base_range=(-5e-3, 5e-3), decay_lambda_base=1e-10,
                 equilibrium_threshold=0.05, entropy_rate=1e-5, zero_replacement_mode=False):
        self.min_tension = min_tension  # Higher for bold pruning like benchmark
        self.base_range = base_range
        self.decay_lambda_base = decay_lambda_base
        self.equilibrium_threshold = equilibrium_threshold  # Tuned for ~40% sparsity
        self.entropy_rate = entropy_rate
        self.zero_replacement_mode = zero_replacement_mode  # Off for pure zeros
        self.learning_rate = 0.001

    def compute_equilibrium(self, input_tensor):
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        device = input_tensor.device
        
        # Mild controlled noise
        mean_abs = torch.mean(torch.abs(input_tensor))
        noise = torch.randn_like(input_tensor) * self.min_tension * 0.5
        output = input_tensor + noise * mean_abs
        
        # Enforce min tension
        near_zero_mask = torch.abs(output) < self.equilibrium_threshold
        if near_zero_mask.any():
            signs = torch.sign(output[near_zero_mask])
            signs[signs == 0] = torch.sign(torch.randn_like(signs))
            output[near_zero_mask] = signs * self.min_tension
        
        # Optional zero replacement (off by default)
        if self.zero_replacement_mode:
            zero_mask = output == 0
            if zero_mask.any():
                output[zero_mask] = torch.randn_like(output[zero_mask]) * self.min_tension
        
        return torch.clamp(output, self.base_range[0], self.base_range[1])

    def hybrid_de_tension_vectorized(self, position_ratios, t=0.0):
        if not isinstance(position_ratios, torch.Tensor):
            position_ratios = torch.tensor(position_ratios, dtype=torch.float32)
        device = position_ratios.device
        
        base_value = torch.exp(-self.decay_lambda_base * t) - self.entropy_rate * t
        tension = torch.full_like(position_ratios, base_value)
        tension = torch.clamp(tension, self.base_range[0], self.base_range[1])
        
        neg_mask = tension < 0
        tension[neg_mask] *= (1 + self.decay_lambda_base * t)
        
        return tension

    def entropy_decay(self, value, t, observer_cost=False):
        decay = value * torch.exp(-self.entropy_rate * t)
        if observer_cost:
            decay -= self.min_tension * t
        return torch.clamp(decay, self.base_range[0], self.base_range[1])

class CurvatureTuner(nn.Module):
    """
    Core accelerated layer: Pruning + deferral + optional zoom.
    """
    def __init__(self, in_features, out_features, framework, freq_factor=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.framework = framework
        self.freq_factor = freq_factor
        self.defer_threshold = framework.min_tension * 15  # Balanced deferral
        self.aggressive_factor = 1.5

    def variable_zoom(self, x, zoom_level=0.0):  # Optional, default no zoom
        if zoom_level != 0.0:
            scale = torch.clamp(torch.tensor(1.0 + zoom_level), 0.1, 10.0)
            x = x * scale * self.freq_factor
        return x

    def forward(self, x, vib_speed=None, zoom_level=0.0):
        x = self.variable_zoom(x, zoom_level)
        
        # Layer norm for stability
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        
        # Per-sample or batch vib_speed
        if vib_speed is None:
            vib_speed = x.norm(dim=-1).mean() if x.dim() > 1 else x.norm()
        
        if vib_speed < self.defer_threshold:
            return self.framework.compute_equilibrium(torch.zeros_like(self.linear(x)))
        
        eq_x = self.framework.compute_equilibrium(x)
        return self.linear(eq_x)

    def tension_accelerated_prune(self, position_ratios=None, t=0.0, aggressive_factor=None, prune_to_zero=True, prune_rate=0.4):
        device = next(self.parameters()).device
        param_scale = sum(p.numel() for p in self.parameters()) / 1e6
        self.aggressive_factor = aggressive_factor or (1.5 + param_scale * 0.2)  # More aggressive for larger
        
        if position_ratios is None:
            num_params = len(list(self.parameters()))
            position_ratios = torch.linspace(0.1, 0.9, num_params, device=device)
        
        tensions = self.framework.hybrid_de_tension_vectorized(position_ratios, t=t)
        effective_before = self.get_effective_params()
        
        with torch.no_grad():
            param_idx = 0
            for param in self.parameters():
                thresh = torch.abs(tensions[param_idx]) * self.framework.min_tension * 20
                if tensions[param_idx] < 0:
                    thresh *= self.aggressive_factor
                mask = torch.abs(param.data) < thresh
                if prune_to_zero:
                    param.data[mask] = 0.0
                else:
                    signs = torch.sign(param.data[mask])
                    signs[signs == 0] = torch.sign(torch.randn_like(signs))
                    param.data[mask] = signs * self.framework.min_tension
                param_idx += 1
        
        effective_after = self.get_effective_params()
        pruned_ratio = (effective_before - effective_after) / max(effective_before, 1)
        
        if pruned_ratio > prune_rate:
            excess = int((pruned_ratio - prune_rate) * effective_before)
            zero_locs = []
            for param in self.parameters():
                zero_mask = (param.data == 0)
                indices = zero_mask.nonzero(as_tuple=False)
                for idx in indices:
                    zero_locs.append((param, tuple(idx.tolist())))
            
            if len(zero_locs) > excess:
                rollback_idx = torch.randperm(len(zero_locs))[:excess]
                for i in rollback_idx:
                    param, idx = zero_locs[i]
                    param.data[idx] = self.framework.min_tension * torch.sign(torch.randn(1, device=device))

    def get_effective_params(self):
        count = 0
        for param in self.parameters():
            count += (torch.abs(param.data) > self.framework.min_tension).sum().item()
        return max(count, 1)

def to_sparse_model(model):
    """Convert to sparse layout for GPU FLOPs savings after pruning."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if module.weight.is_sparse: continue
            module.weight = nn.Parameter(module.weight.data.to_sparse())
            if module.bias is not None:
                module.bias = nn.Parameter(module.bias.data.to_sparse())
    return model

# Supporting classes kept minimal; QuantumBio/MultiObserver/Cosmo can be re-added if needed.
# For training loop example: After every few epochs, call layer.tension_accelerated_prune(t=epoch)
# For max inference speed: model = to_sparse_model(model)
