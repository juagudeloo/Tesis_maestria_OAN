"""
GradNorm: Gradient Normalization for Multi-Task Learning
=========================================================

Implementation of GradNorm algorithm from:
"GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
Chen et al., ICML 2018
https://arxiv.org/abs/1711.02257

Automatically balances gradient magnitudes across multiple loss terms.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


def compute_grad_norm(model: nn.Module) -> float:
    """
    Compute average gradient norm across all model parameters.
    
    Parameters
    ----------
    model : nn.Module
        Model with computed gradients
        
    Returns
    -------
    float
        Average L2 norm of gradients
    """
    total_norm = 0.0
    num_params = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            num_params += 1
    
    if num_params == 0:
        return 0.0
    
    return (total_norm ** 0.5) / num_params


def log_gradient_norms_by_task(
    model: nn.Module,
    losses: Dict[str, torch.Tensor],
    retain_graph: bool = True
) -> Dict[str, float]:
    """
    Compute gradient norms for each task independently.
    
    Parameters
    ----------
    model : nn.Module
        Model to compute gradients for
    losses : Dict[str, torch.Tensor]
        Dictionary of individual loss terms
    retain_graph : bool
        Whether to retain computation graph
        
    Returns
    -------
    Dict[str, float]
        Gradient norm for each task
    """
    grad_norms = {}
    
    for task_name, loss in losses.items():
        model.zero_grad()
        loss.backward(retain_graph=retain_graph)
        grad_norms[task_name] = compute_grad_norm(model)
    
    model.zero_grad()  # Clean up
    return grad_norms


class GradNormScheduler:
    """
    GradNorm scheduler for automatic multi-task loss balancing.
    
    Balances gradient magnitudes across multiple loss terms by learning
    adaptive weights that equalize training rates.
    
    Parameters
    ----------
    num_tasks : int
        Number of loss terms (e.g., 4 for MSE + WFA + Doppler + Temp)
    alpha : float
        Restoring force hyperparameter (typical: 1.0-2.0)
        - alpha=0: Equal gradient magnitudes
        - alpha>0: Allow slower tasks to progress faster
    initial_weights : Optional[List[float]]
        Initial task weights (default: all 1.0)
    device : str
        Device for tensors ('cuda' or 'cpu')
    
    Attributes
    ----------
    task_weights : torch.Tensor
        Learnable weights for each task (shape: num_tasks)
    """
    
    def __init__(
        self,
        num_tasks: int = 4,
        alpha: float = 1.5,
        initial_weights: Optional[List[float]] = None,
        device: str = 'cuda'
    ):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.device = device
        
        # Initialize task weights
        if initial_weights is None:
            initial_weights = [1.0] * num_tasks
        
        self.task_weights = torch.tensor(
            initial_weights,
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )
        
        # Moving averages for gradient norms
        self.avg_grad_norm = torch.ones(num_tasks, device=device)
        
        # Initial loss values (for computing training rate)
        self.initial_losses = None
        
        # Optimizer for task weights
        self.weight_optimizer = torch.optim.Adam([self.task_weights], lr=0.025)
        
        # Tracking
        self.step_count = 0
    
    def compute_weighted_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted combination of losses.
        
        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            Dictionary of loss terms (keys must match task order)
            
        Returns
        -------
        torch.Tensor
            Weighted total loss
        """
        loss_values = list(losses.values())
        weighted_losses = [w * loss for w, loss in zip(self.task_weights, loss_values)]
        return sum(weighted_losses)
    
    def step(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        current_losses: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Update task weights using GradNorm algorithm.
        
        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            Dictionary of individual loss terms (with computational graphs)
        model : nn.Module
            Model with parameters
        current_losses : Optional[Dict[str, float]]
            Current loss values for computing training rate
            
        Returns
        -------
        Dict[str, float]
            Diagnostic metrics (grad_norm_loss, weight updates, etc.)
        """
        self.step_count += 1
        
        # Store initial losses on first call
        if self.initial_losses is None and current_losses is not None:
            self.initial_losses = {k: v for k, v in current_losses.items()}
        
        # 1. Compute gradient norm for each task
        task_names = list(losses.keys())
        G_i = []
        
        for task_name, loss in losses.items():
            model.zero_grad()
            loss.backward(retain_graph=True)
            grad_norm = compute_grad_norm(model)
            G_i.append(grad_norm)
        
        G_i = torch.tensor(G_i, device=self.device)
        
        # 2. Compute relative training rates r_i(t)
        if current_losses is not None and self.initial_losses is not None:
            # r_i(t) = L_i(t) / L_i(0)
            rates = torch.tensor([
                current_losses[task] / (self.initial_losses[task] + 1e-8)
                for task in task_names
            ], device=self.device)
            
            # Average training rate
            avg_rate = rates.mean()
            
            # Inverse relative training rate: r̃_i(t) = r_i(t) / mean(r_j(t))
            relative_rates = rates / (avg_rate + 1e-8)
        else:
            # Fallback: assume equal rates
            relative_rates = torch.ones(self.num_tasks, device=self.device)
        
        # 3. Compute target gradient norms
        # G̅(t) = average gradient norm
        avg_G = G_i.mean()
        
        # Target: G̅(t) × [r̃_i(t)]^alpha
        target_norms = avg_G * (relative_rates ** self.alpha)
        
        # 4. GradNorm loss: L_grad = Σ|G_i(t) - target_i(t)|
        grad_norm_loss = torch.abs(G_i - target_norms).sum()
        
        # 5. Update task weights
        self.weight_optimizer.zero_grad()
        grad_norm_loss.backward()
        self.weight_optimizer.step()
        
        # 6. Normalize weights to sum to num_tasks (preserve overall scale)
        with torch.no_grad():
            # Clamp to prevent extreme values
            self.task_weights.data = torch.clamp(self.task_weights, min=0.01, max=10.0)
            
            # Normalize
            weight_sum = self.task_weights.sum()
            self.task_weights.data = self.task_weights * (self.num_tasks / weight_sum)
        
        # 7. Update moving average
        with torch.no_grad():
            self.avg_grad_norm = 0.9 * self.avg_grad_norm + 0.1 * G_i
        
        # Clean up
        model.zero_grad()
        
        # Return diagnostics
        diagnostics = {
            'grad_norm_loss': grad_norm_loss.item(),
            'avg_grad_norm': avg_G.item(),
        }
        
        # Add per-task metrics
        for i, task_name in enumerate(task_names):
            diagnostics[f'{task_name}_grad_norm'] = G_i[i].item()
            diagnostics[f'{task_name}_weight'] = self.task_weights[i].item()
            diagnostics[f'{task_name}_target_norm'] = target_norms[i].item()
        
        return diagnostics
    
    def get_weights(self) -> Dict[str, float]:
        """Get current task weights as dictionary."""
        return {
            f'weight_{i}': w.item()
            for i, w in enumerate(self.task_weights)
        }
    
    def state_dict(self) -> dict:
        """Return state dictionary for checkpointing."""
        return {
            'task_weights': self.task_weights.detach().cpu(),
            'avg_grad_norm': self.avg_grad_norm.detach().cpu(),
            'initial_losses': self.initial_losses,
            'step_count': self.step_count,
            'alpha': self.alpha,
            'num_tasks': self.num_tasks,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state from checkpoint."""
        self.task_weights = state_dict['task_weights'].to(self.device)
        self.task_weights.requires_grad = True
        self.avg_grad_norm = state_dict['avg_grad_norm'].to(self.device)
        self.initial_losses = state_dict['initial_losses']
        self.step_count = state_dict['step_count']
        self.alpha = state_dict['alpha']
        self.num_tasks = state_dict['num_tasks']
        
        # Recreate optimizer
        self.weight_optimizer = torch.optim.Adam([self.task_weights], lr=0.025)
