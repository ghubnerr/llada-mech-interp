"""
Sparse Autoencoder (SAE) implementations for LLaDA diffusion language models.

Tensor Shape Conventions:
- B: batch size
- T: sequence length (number of tokens)
- N: total tokens (N = B * T for flattened format)
- D: model dimension (hidden size)
- S: number of diffusion steps
- L: number of transformer layers
- K: sparsity parameter (number of active features)
- F: SAE feature dimension (typically F >> D for overcomplete representation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class LLaDASAE(nn.Module):
    """
    k-Sparse Autoencoder (k-SAE) for LLaDA models.

    Architecture:
        Input: x with shape (B, T, D) or (N, D)
        Encoder: W_enc with shape (D, F), b_enc with shape (F)
        Pre-activations: z = xW_enc + b_enc with shape (B, T, F) or (N, F)
        k-Sparsity: a = TopK(z) with exactly K non-zeros per token position
        Decoder: W_dec with shape (F, D), b_dec with shape (D)
        Reconstruction: x_hat = aW_dec + b_dec with same shape as input

    Loss: L = reconstruction_loss + l2_coeff * weight_decay (no L1 penalty needed for k-SAE)
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        k_sparse: int = 64,
        tie_weights: bool = False,
        normalize_decoder: bool = True,
        bias_decoder: bool = True,
        l2_coefficient: float = 1e-6,
        normalize_activations: bool = False,
        activation_norm_eps: float = 1e-6,
    ):
        """
        Initialize LLaDA k-SAE (Top-K Sparse Autoencoder).

        Args:
            d_model (int): Input dimension D (model hidden size)
            d_sae (int): SAE feature dimension F (typically 4-8x larger than d_model)
            k_sparse (int): Number of active features K (k-sparse constraint)
            tie_weights (bool): Whether to tie encoder/decoder weights (W_dec = W_enc^T)
            normalize_decoder (bool): Whether to normalize decoder weights to unit norm
            bias_decoder (bool): Whether to include decoder bias
            l2_coefficient (float): L2 weight decay coefficient
            normalize_activations (bool): Whether to normalize input activations (default: False for backward compatibility)
            activation_norm_eps (float): Epsilon for activation normalization stability
        """
        super().__init__()

        # Store configuration
        self.d_model = d_model  # D
        self.d_sae = d_sae  # F
        self.k_sparse = k_sparse  # K
        self.tie_weights = tie_weights
        self.normalize_decoder = normalize_decoder
        self.l2_coefficient = l2_coefficient
        self.normalize_activations = normalize_activations
        self.activation_norm_eps = activation_norm_eps

        self.encoder = nn.Linear(d_model, d_sae, bias=True)

        if tie_weights:
            # Tied weights: W_dec = W_enc.T, so no separate decoder weight matrix
            self.decoder_bias = (
                nn.Parameter(torch.zeros(d_model)) if bias_decoder else None
            )
        else:
            self.decoder = nn.Linear(d_sae, d_model, bias=bias_decoder)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        # Encoder initialization: Xavier uniform
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        if not self.tie_weights:
            # Decoder initialization: Xavier uniform
            nn.init.xavier_uniform_(self.decoder.weight)
            if self.decoder.bias is not None:
                nn.init.zeros_(self.decoder.bias)
        else:
            if self.decoder_bias is not None:
                nn.init.zeros_(self.decoder_bias)

    def _normalize_decoder_weights(self):
        """Normalize decoder weight columns to unit norm (optional constraint)."""
        if self.normalize_decoder:
            if self.tie_weights:
                # For tied weights, normalize encoder weight rows (which become decoder columns)
                with torch.no_grad():
                    self.encoder.weight.data = F.normalize(
                        self.encoder.weight.data, dim=0
                    )
            else:
                # Normalize decoder weight columns
                with torch.no_grad():
                    self.decoder.weight.data = F.normalize(
                        self.decoder.weight.data, dim=1
                    )

    def _normalize_activations(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize input activations to have zero mean and unit variance.

        This helps stabilize training by ensuring activations are in a consistent scale,
        which is particularly important for MSE reconstruction loss.

        Args:
            x: Input activations (..., D)

        Returns:
            normalized_x: Normalized activations (..., D)
            mean: Mean values used for normalization (..., D)
            std: Standard deviation values used for normalization (..., D)
        """
        if not self.normalize_activations:
            return x, torch.zeros_like(x), torch.ones_like(x)

        # Compute statistics along the feature dimension (last dim)
        mean = x.mean(dim=-1, keepdim=True)  # (..., 1)
        std = (
            x.std(dim=-1, keepdim=True, unbiased=False) + self.activation_norm_eps
        )  # (..., 1)

        # Normalize: (x - mean) / std
        normalized_x = (x - mean) / std

        return normalized_x, mean.squeeze(-1), std.squeeze(-1)

    def _denormalize_activations(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """
        Denormalize activations back to original scale.

        Args:
            x: Normalized activations (..., D)
            mean: Mean values from normalization (...,) or (..., D)
            std: Standard deviation values from normalization (...,) or (..., D)

        Returns:
            denormalized_x: Activations in original scale (..., D)
        """
        if not self.normalize_activations:
            return x

        # Ensure mean and std have the right shape for broadcasting
        if mean.dim() == x.dim() - 1:
            mean = mean.unsqueeze(-1)  # (..., 1)
        if std.dim() == x.dim() - 1:
            std = std.unsqueeze(-1)  # (..., 1)

        # Denormalize: x * std + mean
        return x * std + mean

    def _apply_k_sparse(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """
        Apply k-sparse (Top-K) constraint to pre-activations.

        Args:
            pre_acts: Pre-activation tensor z
                     Supported shapes:
                     - (B, T, F): batch_size x sequence_length x features
                     - (N, F): total_tokens x features (flattened format)

        Returns:
            acts: k-sparse activation tensor a (same shape as input)

        k-Sparse Method:
        Keep only top K activations per token position:
        For 3D: a[b,t,f] = z[b,t,f] if f in TopK(z[b,t,:]), else 0
        For 2D: a[n,f] = z[n,f] if f in TopK(z[n,:]), else 0
        """
        # Top-K sparsity: keep only K largest activations per token position
        # Works for both (B, T, F) and (N, F) input shapes

        acts = torch.zeros_like(pre_acts)  # Initialize sparse tensor

        # Get top-k values and indices along the feature dimension (last dim)
        # For (B, T, F): topk_vals: (B, T, K), topk_indices: (B, T, K)
        # For (N, F): topk_vals: (N, K), topk_indices: (N, K)
        topk_vals, topk_indices = torch.topk(pre_acts, self.k_sparse, dim=-1)

        # Scatter top-k values back to full tensor
        # For 3D: acts[b, t, topk_indices[b, t, k]] = topk_vals[b, t, k]
        # For 2D: acts[n, topk_indices[n, k]] = topk_vals[n, k]
        acts.scatter_(-1, topk_indices, topk_vals)

        return acts

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SAE.

        Mathematical formulation:
        1. Normalize input: x_norm = (x - mean) / std
        2. Pre-activations: z = x_norm*W_enc + b_enc
        3. k-Sparse activations: a = TopK(z)
        4. Reconstruction: x_hat_norm = aW_dec + b_dec
        5. Denormalize: x_hat = x_hat_norm * std + mean

        Args:
            x: Input activations from LLaDA model
               Supported shapes:
               - (B, T, D): batch_size x sequence_length x model_dimension
               - (N, D): total_tokens x model_dimension (flattened format)
               where D = model dimension (d_model)

        Returns:
            reconstruction: Reconstructed activations x_hat (same shape as input)
            sparse_acts: Sparse feature activations a
                        Shape: (..., F) where F = d_sae
            pre_acts: Pre-activation values z (before sparsity)
                     Shape: (..., F) - useful for analysis

        Tensor Flow:
        x: (..., D) -> [Normalize] -> x_norm: (..., D) -> [Linear] -> z: (..., F) -> [TopK] ->
        a: (..., F) -> [Linear] -> x_hat_norm: (..., D) -> [Denormalize] -> x_hat: (..., D)
        """
        # Step 1: Normalize input activations
        x_norm, x_mean, x_std = self._normalize_activations(x)  # (..., D), (...), (...)

        # Step 2: Encode to feature space
        # z = x_norm*W_enc + b_enc
        # x_norm: (..., D) @ W_enc: (D, F) + b_enc: (F,) -> z: (..., F)
        pre_acts = self.encoder(x_norm)  # (..., F)

        # Step 3: Apply k-sparse constraint
        # a = TopK(z) -> (..., F) with exactly K non-zero elements per token
        sparse_acts = self._apply_k_sparse(pre_acts)  # (..., F)

        # Step 4: Decode back to normalized model space
        if self.tie_weights:
            # Tied weights: use transposed encoder weights as decoder
            # x_hat_norm = aW_enc.T + b_dec
            # a: (..., F) @ W_enc.T: (F, D) + b_dec: (D,) -> x_hat_norm: (..., D)
            reconstruction_norm = F.linear(
                sparse_acts, self.encoder.weight.t(), self.decoder_bias
            )
        else:
            # Separate decoder weights
            # x_hat_norm = aW_dec + b_dec
            # a: (..., F) @ W_dec: (F, D) + b_dec: (D,) -> x_hat_norm: (..., D)
            reconstruction_norm = self.decoder(sparse_acts)  # (..., D)

        # Step 5: Denormalize reconstruction back to original scale
        reconstruction = self._denormalize_activations(
            reconstruction_norm, x_mean, x_std
        )

        # Normalize decoder weights if requested (maintains unit norm constraint)
        if self.training and self.normalize_decoder:
            self._normalize_decoder_weights()

        return reconstruction, sparse_acts, pre_acts

    @staticmethod
    def flatten_sequence_tensor(
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Flatten (batch_size, sequence_length, hidden_size) to (total_tokens, hidden_size).

        Args:
            x: Input tensor with shape (B, T, D)

        Returns:
            flattened: Tensor with shape (B*T, D)
            original_shape: Tuple (B, T) for reconstruction
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B, T, D), got {x.dim()}D tensor")

        batch_size, seq_length, hidden_size = x.shape
        flattened = x.view(batch_size * seq_length, hidden_size)
        return flattened, (batch_size, seq_length)

    @staticmethod
    def unflatten_sequence_tensor(
        x: torch.Tensor, original_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Unflatten (total_tokens, hidden_size) back to (batch_size, sequence_length, hidden_size).

        Args:
            x: Flattened tensor with shape (B*T, D)
            original_shape: Tuple (B, T) from flatten_sequence_tensor

        Returns:
            unflattened: Tensor with shape (B, T, D)
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D tensor (N, D), got {x.dim()}D tensor")

        batch_size, seq_length = original_shape
        total_tokens, hidden_size = x.shape

        if total_tokens != batch_size * seq_length:
            raise ValueError(
                f"Shape mismatch: expected {batch_size * seq_length} tokens, "
                f"got {total_tokens}"
            )

        return x.view(batch_size, seq_length, hidden_size)

    def forward_flattened(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with automatic flattening for (B, T, D) inputs.

        This method automatically flattens 3D inputs to 2D, processes them,
        and reshapes outputs back to 3D format. Useful for maintaining
        backward compatibility with existing code.

        Args:
            x: Input tensor with shape (B, T, D)

        Returns:
            reconstruction: Reconstructed tensor with shape (B, T, D)
            sparse_acts: Sparse activations with shape (B, T, F)
            pre_acts: Pre-activations with shape (B, T, F)
        """
        if x.dim() == 2:
            # Already flattened, use regular forward pass
            return self.forward(x)
        elif x.dim() == 3:
            # Flatten, process, and unflatten
            x_flat, original_shape = self.flatten_sequence_tensor(x)
            recon_flat, sparse_flat, pre_flat = self.forward(x_flat)

            reconstruction = self.unflatten_sequence_tensor(recon_flat, original_shape)
            sparse_acts = self.unflatten_sequence_tensor(sparse_flat, original_shape)
            pre_acts = self.unflatten_sequence_tensor(pre_flat, original_shape)

            return reconstruction, sparse_acts, pre_acts
        else:
            raise ValueError(f"Expected 2D or 3D input tensor, got {x.dim()}D")

    def compute_loss(
        self, x: torch.Tensor, reconstruction: torch.Tensor, sparse_acts: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SAE training losses.

        Total Loss: L = L_recon + l2_coeff * L_weight_decay

        Args:
            x: Original input
               Supported shapes: (B, T, D) or (N, D)
            reconstruction: Reconstructed input (same shape as x)
            sparse_acts: Sparse activations
                        Shapes: (B, T, F) or (N, F)

        Returns:
            loss_dict: Dictionary containing:
                - total_loss: Combined loss for backpropagation
                - recon_loss: Reconstruction loss (squared error)
                - weight_decay_loss: L2 weight regularization
                - metrics: Additional metrics (sparsity ratio, etc.)
        """
        # Reconstruction loss: MSE between input and reconstruction
        # L_recon = mean_squared_error(x, x_hat)
        recon_loss = F.mse_loss(reconstruction, x)

        # Weight decay loss: L2 penalty on parameters
        weight_decay_loss = 0.0
        if self.tie_weights:
            weight_decay_loss += torch.norm(self.encoder.weight) ** 2
        else:
            weight_decay_loss += torch.norm(self.decoder.weight) ** 2
            weight_decay_loss += torch.norm(self.encoder.weight) ** 2

        # Total loss combination (no L1 sparsity needed for k-SAE)
        total_loss = recon_loss + self.l2_coefficient * weight_decay_loss

        # Compute metrics
        with torch.no_grad():
            # Sparsity ratio: fraction of non-zero activations
            sparsity_ratio = (sparse_acts != 0).float().mean()

            # Average L0 norm (number of active features per token)
            l0_norm = (sparse_acts != 0).sum(dim=-1).float().mean()

            # Reconstruction error statistics
            # For both (B, T, D) and (N, D) inputs, compute L2 norm along last dimension
            recon_error = torch.norm(x - reconstruction, dim=-1)  # (B, T) or (N,)
            max_recon_error = recon_error.max()
            mean_recon_error = recon_error.mean()

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "weight_decay_loss": weight_decay_loss,
            "sparsity_ratio": sparsity_ratio,
            "l0_norm": l0_norm,
            "max_recon_error": max_recon_error,
            "mean_recon_error": mean_recon_error,
        }

    def get_feature_activations(
        self, x: torch.Tensor, return_indices: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get sparse feature activations for analysis.

        Args:
            x: Input tensor
               Supported shapes: (B, T, D) or (N, D)
            return_indices: Whether to return indices of active features

        Returns:
            sparse_acts: Sparse activations
                        Shapes: (B, T, F) or (N, F)
            active_indices: Indices of active features
                           Shapes: (B, T, K) or (N, K) if return_indices=True
        """
        with torch.no_grad():
            _, sparse_acts, _ = self.forward(x)

            if return_indices:
                pre_acts = self.encoder(x)
                _, active_indices = torch.topk(pre_acts, self.k_sparse, dim=-1)
                return sparse_acts, active_indices

            return sparse_acts, None
