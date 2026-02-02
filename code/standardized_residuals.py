import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import copy

def get_volumes(Sigma, radius):
        """
        Calculates the volume of the set {y | sqrt(y^T Sigma^-1 y) <= radius}.
        
        Args:
            Sigma: Covariance matrix (B, K, K) or (K, K).
            radius: The threshold value for the squared Mahalanobis distance (scalar or (B,)).
                    Note: This is 'r' in the inequality, equivalent to the squared geometric radius.
        
        Returns:
            volume: Tensor of volumes.
        """
        radius_squared = radius**2
        # Handle batch dimensions
        if Sigma.dim() == 2:
            Sigma = Sigma.unsqueeze(0)
        
        B, K, _ = Sigma.shape
        device = Sigma.device
        
        # Ensure radius is a tensor for broadcasting
        if isinstance(radius_squared, (int, float)):
            radius_squared = torch.tensor(radius_squared, device=device)
        if radius_squared.dim() == 0:
            radius_squared = radius_squared.expand(B)
            
        # Compute Log Determinant of Sigma
        # We use Cholesky for stability: det(Sigma) = det(L)^2 -> log(det) = 2 * sum(log(diag(L)))
        # Note: Sigma must be positive definite.
        L = torch.linalg.cholesky(Sigma)
        log_det_sigma = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
        
        # Compute Log Volume of Unit Ball (dimension K)
        # Log(pi^(K/2) / Gamma(K/2 + 1)) = (K/2) * log(pi) - lgamma(K/2 + 1)
        k_tensor = torch.tensor(K, dtype=torch.float, device=device)
        log_pi = math.log(math.pi)
        log_unit_ball = (k_tensor / 2) * log_pi - torch.lgamma(k_tensor / 2 + 1)
        
        # Compute Log Radius Term
        # log = (K/2) * log(radius)
        # Clamp radius to positive to avoid log domain errors
        safe_radius = torch.clamp(radius_squared, min=1e-9)
        log_radius_term = (k_tensor / 2) * torch.log(safe_radius)
        
        # Combine and Exponentiate
        # V = sqrt(det(Sigma)) * V_unit * radius^(K/2)
        # log(V) = 0.5 * log_det_sigma + log_unit_ball + log_radius_term
        log_vol = 0.5 * log_det_sigma + log_unit_ball + log_radius_term
        
        return torch.exp(log_vol)



# ==========================================
# The model architecture (Backbone + Head)
# ==========================================

class SimpleTabularMLP(nn.Module):
    """
    Robust Backbone: ResNet-MLP for Tabular Data
    """
    def __init__(self, num_cont, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.first_layer = nn.Linear(num_cont, hidden_dim)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.first_layer(x)
        for block in self.blocks:
            x = x + block(x)  
        return self.norm(x)

class RobustCovarianceHead(nn.Module):
    """
    Unified Head: Switches between Full Cholesky and Low-Rank.
    """
    def __init__(self, input_dim, y_dim, init_sigma=1.0, mode="full_cholesky"):
        super().__init__()
        self.y_dim = y_dim
        if mode == 'low_rank':
            self.mode = 'low_rank'
            rank = int(math.ceil(math.sqrt(y_dim)))
            self.fc_mu = nn.Linear(input_dim, y_dim)
            self.fc_log_diag = nn.Linear(input_dim, y_dim)
            self.fc_factors = nn.Linear(input_dim, y_dim * rank)
            self.rank = rank
        elif mode=="full_cholesky":
            if y_dim > 10: print("Large output dimension, initializing with mode = 'low_rank' is recommanded.")
            self.mode = 'full_cholesky'
            self.fc_mu = nn.Linear(input_dim, y_dim)
            num_chol = (y_dim * (y_dim + 1)) // 2
            self.fc_chol = nn.Linear(input_dim, num_chol)
            self.register_buffer('tril_indices', torch.tril_indices(y_dim, y_dim))
            
            # Initialize diagonal to be positive/stable
            with torch.no_grad():
                diag_mask = (self.tril_indices[0] == self.tril_indices[1])
                inv_softplus = math.log(math.exp(init_sigma) - 1)
                self.fc_chol.bias[diag_mask] = inv_softplus
        else:
            raise ValueError("The mode must either be 'full_cholesky' or 'low_rank'.")

    def forward(self, x):
        if self.mode == 'low_rank':
            B = x.shape[0]
            mu = self.fc_mu(x)
            D = torch.exp(self.fc_log_diag(x)) + 1e-6
            V = self.fc_factors(x).view(B, self.y_dim, self.rank)
            return (mu, D, V)
        else:
            B = x.shape[0]
            mu = self.fc_mu(x)
            chol_flat = self.fc_chol(x)
            L = torch.zeros(B, self.y_dim, self.y_dim, device=x.device)
            L[:, self.tril_indices[0], self.tril_indices[1]] = chol_flat
            # Softplus diagonal
            diag_idx = torch.arange(self.y_dim, device=x.device)
            L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-6
            return (mu, L)

# ==========================================
# The trainer
# ==========================================
           
class Trainer:
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=0.1, init_sigma=1.0, mode="full_cholesky", center_model=None):
        self.backbone = SimpleTabularMLP(num_cont=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.head = RobustCovarianceHead(input_dim=hidden_dim, y_dim=output_dim, init_sigma=init_sigma, mode=mode)
        self.center_model = center_model
        self.y_dim = output_dim
        self.mode = mode

    def forward(self, bx):
        feats = self.backbone(bx)
        preds = self.head(feats)
        if self.center_model is not None:
            with torch.no_grad():
                center_pred = self.center_model(bx)
                preds = list(preds)
                if isinstance(center_pred, torch.Tensor):
                    center_tensor = center_pred
                else:
                    center_tensor = torch.from_numpy(center_pred)
                center_tensor = center_tensor.to(
                    dtype=preds[1].dtype,
                    device=preds[1].device
                )
                preds[0] = center_tensor
                preds = tuple(preds)

        return preds

    def fit(self,
            trainloader,
            stoploader,
            num_epochs=300,
            lr=1e-3,
            weight_decay=1e-4,
            verbose=-1,
            NaN_Values = False
            ):


        optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        best_stop_loss = float("inf")
        best_backbone_state = None
        best_head_state = None

        if verbose == 1:
            print_every = max(1, num_epochs // 10)
        elif verbose == 2:
            print_every = 1
        else:
            print_every = num_epochs + 1

        for epoch in range(1, num_epochs):
            self.backbone.train()
            self.head.train()
            total_loss = 0.0

            for bx, by in trainloader:
                optimizer.zero_grad()

                preds = self.forward(bx)

                if not NaN_Values: loss = self.loss(preds, by)
                else : loss = self.loss_with_missing(preds, by)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(trainloader)
        
            self.backbone.eval()
            self.head.eval()
            total_stop_loss = 0.0

            with torch.no_grad():
                for bx, by in stoploader:
            
                    preds = self.forward(bx)

                    if not NaN_Values: loss = self.loss(preds, by)
                    else : loss = self.loss_with_missing(preds, by)
                    
                    total_stop_loss += loss.item()

                avg_stop_loss = total_stop_loss / len(stoploader)

            if verbose != -1 and epoch % print_every == 0:
                print(f"Epoch {epoch}: Avg NLL Loss = {avg_train_loss:.4f} -- Stop loss: {avg_stop_loss:.4f} -- Best Stop Loss: {best_stop_loss}")
                
            if avg_stop_loss < best_stop_loss:
                best_stop_loss = avg_stop_loss
                best_backbone_state = copy.deepcopy(self.backbone.state_dict())
                best_head_state = copy.deepcopy(self.head.state_dict())
            
        # Restore best model
        self.backbone.load_state_dict(best_backbone_state)
        self.head.load_state_dict(best_head_state)

        if verbose != -1:
            print(f"Best stop loss achieved: {best_stop_loss:.4f}")

    def loss(self, params, y_target):
        if self.head.mode == 'low_rank':
            # Woodbury Identity Loss
            mu, D, V = params
            r = y_target - mu
            B, Y, K = V.shape
            inv_std = 1.0 / D
            W = V * inv_std.unsqueeze(-1)
            z = r * inv_std
            
            M = torch.eye(K, device=V.device).unsqueeze(0) + torch.bmm(W.transpose(1, 2), W)
            L_M = torch.linalg.cholesky(M)
            
            log_det = 2 * torch.sum(torch.log(D), 1) + 2 * torch.sum(torch.log(torch.diagonal(L_M, dim1=-2, dim2=-1)), 1)
            
            z_sq = torch.sum(z**2, 1)
            p = torch.bmm(W.transpose(1, 2), z.unsqueeze(-1))
            q = torch.cholesky_solve(p, L_M)
            quad = torch.bmm(p.transpose(1, 2), q).squeeze()
            
            return 0.5 * (z_sq - quad + log_det).mean()
        
        else:
            # Full Cholesky Loss
            mu, L = params
            diff = (y_target - mu).unsqueeze(-1)
            z = torch.triangular_solve(diff, L, upper=False)[0]
            mahalanobis = torch.sum(z.squeeze(-1) ** 2, dim=1)
            log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1)
            return 0.5 * (mahalanobis + log_det).mean()
        
    def loss_with_missing(self, params, y_target):
        """
        Computes Gaussian NLL while automatically handling missing data (NaNs).
        
        Args:
            params: Tuple of model outputs (depends on mode).
            y_target: Tensor (Batch, Y_dim) containing floats and NaNs.
        """
        # Create Mask (1 = Observed, 0 = Missing)
        mask = (~torch.isnan(y_target)).float()
        
        # Safe Target (Fill NaNs with 0 to prevent NaN propagation in gradients)
        y_safe = torch.nan_to_num(y_target, nan=0.0)

        # ==========================================
        # Strategy A: Low-Rank (Vectorized & Fast)
        # ==========================================
        if self.head.mode == 'low_rank':
            mu, D, V = params
            
            # Residuals (Zero out missing residuals)
            r = (y_safe - mu) * mask 
            
            # Dimensions
            B, Y, K = V.shape
            
            # Inverse Variance (Diagonal)
            # D contains standard deviations. 
            # We set inverse variance to 0 where data is missing (Infinite uncertainty)
            inv_std = 1.0 / (D + 1e-8) # Add tiny epsilon to D to avoid div by zero
            
            # Whiten Factors: W = V * D^-1/2 * mask
            # This mathematically removes rows corresponding to missing outputs
            # from the low-rank update structure.
            W = V * (inv_std * mask).unsqueeze(-1)
            
            # Core Matrix M = I + W^T W (Size K x K)
            M = torch.eye(K, device=V.device).unsqueeze(0) + torch.bmm(W.transpose(1, 2), W)
            L_M = torch.linalg.cholesky(M)
            
            # Log Determinant
            # log|Sigma_obs| = log|D_obs| + log|M|
            log_D = torch.log(D + 1e-8) * mask
            log_det_D = 2 * torch.sum(log_D, dim=1)
            
            # log|M| = 2 * sum(log(diag(L_M)))
            log_det_M = 2 * torch.sum(torch.log(torch.diagonal(L_M, dim1=-2, dim2=-1)), dim=1)
            
            # Mahalanobis Term using Woodbury Identity
            # z = D^-1/2 * r (masked)
            z = r * inv_std
            z_sq = torch.sum(z**2, dim=1)
            
            # Woodbury correction: p = W^T z
            p = torch.bmm(W.transpose(1, 2), z.unsqueeze(-1))
            
            # Solve M * q = p
            q = torch.cholesky_solve(p, L_M)
            
            # Term: p^T q
            quad = torch.bmm(p.transpose(1, 2), q).squeeze(-1).squeeze(-1)
            
            # Final NLL Calculation
            num_obs = mask.sum(dim=1)
            
            # Avoid calculating loss for samples with 0 observations
            safe_norm = torch.where(num_obs > 0, 1.0, 0.0)
            
            nll = 0.5 * (z_sq - quad + log_det_D + log_det_M + num_obs * math.log(2 * math.pi))
            
            return (nll * safe_norm).mean()

        # ==========================================
        # Strategy B: Full Cholesky (Looped & Robust)
        # ==========================================
        else:
            mu_batch, L_batch = params
            batch_size = mu_batch.shape[0]
            total_loss = 0.0
            valid_samples = 0
            
            # Pre-compute Sigma = L @ L.T
            Sigma_batch = torch.bmm(L_batch, L_batch.transpose(1, 2))
            
            for i in range(batch_size):
                # Identify observed indices
                obs_indices = torch.nonzero(mask[i]).squeeze(-1)
                
                # Skip if entire row is missing
                if len(obs_indices) == 0:
                    continue
                
                valid_samples += 1
                
                # Slice Y and Mu
                y_sub = y_safe[i, obs_indices]
                mu_sub = mu_batch[i, obs_indices]
                
                # Slice Sigma (The correct PyTorch way)
                # Select rows first, then columns
                Sigma_sub = Sigma_batch[i][obs_indices][:, obs_indices]
                
                # Re-Cholesky the submatrix
                # We add a tiny jitter (1e-6) to ensure positive definiteness 
                # is maintained after slicing and floating point noise
                L_sub = torch.linalg.cholesky(Sigma_sub + 1e-6 * torch.eye(len(obs_indices), device=y_target.device))
                
                # Standard Gaussian Loss
                diff = (y_sub - mu_sub).unsqueeze(-1)
                
                # Solve L * z = diff
                z = torch.triangular_solve(diff, L_sub, upper=False)[0]
                
                mahalanobis = torch.sum(z.squeeze(-1) ** 2)
                log_det = 2 * torch.sum(torch.log(torch.diagonal(L_sub)))
                
                loss_i = 0.5 * (mahalanobis + log_det + len(obs_indices) * math.log(2 * math.pi))
                total_loss += loss_i
            
            if valid_samples == 0:
                return torch.tensor(0.0, device=y_target.device, requires_grad=True)
                
            return total_loss / valid_samples 
    
# ==========================================
# The conformal class manager
# ==========================================

class StandardizedResiduals():
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=0.1, init_sigma=1.0, center_model=None, mode="full_cholesky"):
        self.trainer = Trainer(input_dim, 
                            output_dim,
                            hidden_dim=hidden_dim, 
                            num_layers=num_layers, 
                            dropout=dropout, 
                            init_sigma=init_sigma, 
                            center_model=center_model,
                            mode = mode
                            )
        
        self.y_dim = output_dim

    def fit(self,
            trainloader,
            stoploader,
            num_epochs=300,
            lr=1e-3,
            weight_decay=1e-4,
            verbose=-1,
            NaN_Values = False
            ):


        self.trainer.fit(trainloader,
            stoploader,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            verbose=verbose,
            NaN_Values = NaN_Values
            )


    def forward(self, x):
        return self.trainer.forward(x)

    def _compute_mahalanobis_low_rank(self, y, mu, D, V):
        """
        Helper: Computes sqrt( (y-mu)^T Sigma^-1 (y-mu) ) using Woodbury Identity.
        """
        r = y - mu
        inv_std = 1.0 / D
        W = V * inv_std.unsqueeze(-1)  # (B, Y, Rank)
        z = r * inv_std                # (B, Y)

        # M = I + W^T W
        B, Y, K = V.shape
        M = torch.eye(K, device=V.device).unsqueeze(0) + torch.bmm(W.transpose(1, 2), W)
        L_M = torch.linalg.cholesky(M)

        # Term 1: z^T z
        z_sq = torch.sum(z**2, dim=1)

        # Term 2: z^T V (I + V^T D^-2 V)^-1 V^T z
        # Let p = W^T z
        p = torch.bmm(W.transpose(1, 2), z.unsqueeze(-1)) # (B, Rank, 1)
        
        # Solve M q = p  => q = M^-1 p
        q = torch.cholesky_solve(p, L_M)
        
        # quad = p^T q
        quad = torch.bmm(p.transpose(1, 2), q).squeeze(-1).squeeze(-1)
        
        return torch.sqrt(z_sq - quad)

    def _compute_mahalanobis_full_chol(self, y, mu, L):
        """
        Helper: Computes (y-mu)^T Sigma^-1 (y-mu) using triangular solve.
        """
        diff = (y - mu).unsqueeze(-1)
        # Solve L z = (y - mu)
        z = torch.triangular_solve(diff, L, upper=False)[0]
        return torch.sqrt(torch.sum(z.squeeze(-1) ** 2, dim=1))

    def get_standardized_score(self, x, y):
        """
        Returns a tensor of shape (len(x),) containing (y-mu)^T Sigma^-1 (y-mu).
        """
        params = self.forward(x)
        
        if self.trainer.mode == 'low_rank':
            mu, D, V = params
            return self._compute_mahalanobis_low_rank(y, mu, D, V)
        else:
            mu, L = params
            return self._compute_mahalanobis_full_chol(y, mu, L)

    def get_standardized_missing_score(self, x, y):
        """
        Calculates the Chi-squared CDF of the Mahalanobis distance for the observed (non-NaN) outputs.
        
        Args:
            x: Input tensor (B, input_dim)
            y: Target tensor (B, y_dim) containing NaNs for missing values.
            
        Returns:
            scores: Tensor (B,) between 0 and 1 representing the CDF value.
        """
        B = x.shape[0]
        device = x.device
        
        # Get Distribution Parameters
        params = self.forward(x)
        
        # Identify Observed Masks
        # mask is True where y is observed (not NaN)
        mask = ~torch.isnan(y)
        
        # Check if we can use the fast vectorized path (all values observed)
        if mask.all():
            # Use the efficient full-batch implementation we already have
            if self.trainer.mode == 'low_rank':
                mu, D, V = params
                d2 = self._compute_mahalanobis_low_rank(y, mu, D, V)
            else:
                mu, L = params
                d2 = self._compute_mahalanobis_full_chol(y, mu, L)
            
            # CDF(d^2, df=y_dim)
            k = self.y_dim
            return torch.special.gammainc(torch.tensor(k / 2.0, device=device), d2 / 2.0)

        # Handle Mixed Missingness (Per-Sample Loop)
        # Because every sample might have a different sparsity pattern, we must 
        # iterate or group. For robustness, we iterate.
        
        scores = torch.zeros(B, device=device)
        
        # Pre-process parameters for easier indexing
        if self.trainer.mode == 'low_rank':
            mu_batch, D_batch, V_batch = params
            # Pre-calculate Full Sigma for slicing if simple low rank
            # Note: For massive dims, one might want to keep using Woodbury per slice, 
            # but usually slicing the dense reconstruction is faster for "output_dim ~ 100s"
            # Sigma = D + V @ V^T
            # We reconstruct per row inside loop to save memory or reconstruct once if small.
        else:
            mu_batch, L_batch = params
            # Sigma = L @ L^T
        
        for i in range(B):
            # Indices of observed values for this sample
            idx_obs = torch.where(mask[i])[0]
            k = len(idx_obs)
            
            if k == 0:
                # No observations -> Score is 0 (or undefined). 
                scores[i] = 0.0
                continue
            
            # Extract observed values
            y_sub = y[i, idx_obs]
            mu_sub = mu_batch[i, idx_obs]
            
            # Extract observed Covariance Sigma_sub
            if self.trainer.mode == 'low_rank':
                # Sigma_sub = diag(D[idx]) + V[idx] @ V[idx].T
                D_sub = D_batch[i, idx_obs]
                V_sub = V_batch[i, idx_obs, :] # (k, rank)
                Sigma_sub = torch.matmul(V_sub, V_sub.T)
                # Add diagonal efficiently
                Sigma_sub.diagonal().add_(D_sub)
            else:
                # Reconstruct full Sigma for this sample then slice
                # (Doing this per sample avoids B*Y*Y memory allocation)
                L_i = L_batch[i]
                Sigma_full = torch.matmul(L_i, L_i.T)
                # Slice the k x k block
                Sigma_sub = Sigma_full[idx_obs][:, idx_obs]

            # Compute Mahalanobis: (y - mu)^T * Sigma^-1 * (y - mu)
            # Use Cholesky solve for stability on the sub-matrix
            # Sigma_sub is symmetric positive definite
            
            # Add tiny jitter to prevent numerical issues with slices of low-rank matrices
            Sigma_sub.diagonal().add_(1e-6)
            
            L_sub = torch.linalg.cholesky(Sigma_sub)
            diff = (y_sub - mu_sub).unsqueeze(-1) # (k, 1)
            
            # Solve L z = diff
            z = torch.triangular_solve(diff, L_sub, upper=False)[0]
            d2_i = torch.sum(z**2)
            
            # Compute Chi2 CDF
            # P(k/2, d2/2)
            prob = torch.special.gammainc(torch.tensor(k / 2.0, device=device), d2_i / 2.0)
            scores[i] = prob
            
        return scores

    def get_projection_standardized_score(self, projection_matrix, x, y):
        """
        Calculates Mahalanobis distance for the distribution of (projection_matrix @ Y) | X.
        projection_matrix: Shape (K, y_dim) or (B, K, y_dim)
        """
        params = self.forward(x)
        
        # Ensure projection matrix matches batch dimensions if necessary
        # We assume standard case where projection_matrix is (K, y_dim) shared across batch
        if projection_matrix.dim() == 2:
            P = projection_matrix.unsqueeze(0).expand(x.shape[0], -1, -1) # (B, K, Y)
        else:
            P = projection_matrix
            
        y_proj = torch.bmm(P, y.unsqueeze(-1)).squeeze(-1) # (B, K)

        if self.trainer.mode == 'low_rank':
            mu, D, V = params
            
            # projection Mean: P @ mu
            mu_proj = torch.bmm(P, mu.unsqueeze(-1)).squeeze(-1)
            
            # projection Covariance construction:
            # Sigma_proj = P (D^2 + V V^T) P^T 
            #            = (P @ diag(D^2) @ P^T) + (P @ V) @ (P @ V)^T
            
            # Diagonal part: P @ diag(D^2) @ P^T
            # Trick: (P * D^2) @ P^T is faster
            D_sq = D ** 2
            P_scaled = P * D_sq.unsqueeze(1) # Broadcast D across rows of P
            term1 = torch.bmm(P_scaled, P.transpose(1, 2))
            
            # Low rank part: (P @ V) @ (V^T @ P^T)
            PV = torch.bmm(P, V)
            term2 = torch.bmm(PV, PV.transpose(1, 2))
            
            Sigma_proj = term1 + term2
            
        else:
            mu, L = params
            mu_proj = torch.bmm(P, mu.unsqueeze(-1)).squeeze(-1)
            
            # Sigma_proj = P (L L^T) P^T = (P L) (P L)^T
            PL = torch.bmm(P, L)
            Sigma_proj = torch.bmm(PL, PL.transpose(1, 2))

        # Add small jitter for stability in projection space
        diag_idx = torch.arange(Sigma_proj.shape[1], device=x.device)
        Sigma_proj[:, diag_idx, diag_idx] += 1e-6

        # Since projection dimension K is usually small, we treat Sigma_proj as a Full Dense Cholesky
        L_proj = torch.linalg.cholesky(Sigma_proj)
        
        return self._compute_mahalanobis_full_chol(y_proj, mu_proj, L_proj)

    def get_revealed_standardized_score(self, idx_revealed, x, y):
        """
        Returns score for Y[idx_hidden] | Y[idx_revealed].
        Math: Score(Joint) - Score(Revealed_Marginal).
        """
        # Total Score (Joint Distribution)
        total_score = self.get_standardized_score(x, y)
        
        # Edge cases
        if len(idx_revealed) == 0:
            return total_score
        if len(idx_revealed) == self.y_dim:
            return torch.zeros_like(total_score)

        # Construct Projection Matrix for Marginal Distribution of Revealed
        # Shape (K, y_dim) where rows are one-hot vectors for revealed indices
        K = len(idx_revealed)
        P = torch.zeros(K, self.y_dim, device=x.device)
        P[range(K), idx_revealed] = 1.0
        
        # Marginal Score
        # We perform P @ Y (which effectively selects y[idx_revealed])
        marginal_score = self.get_projection_standardized_score(P, x, y)
        
        # Conditional Score = Joint - Marginal
        return torch.sqrt( total_score**2 - marginal_score**2 )
    
    def get_distribution(self, x):
        """
        Computes the full distribution of Y | X.
        
        Args:
            x: Input tensor (B, input_dim)
            
        Returns:
            mu: (B, y_dim) - Point predictions (Mean)
            Sigma: (B, y_dim, y_dim) - Full covariance matrix 
        """
        params = self.forward(x)
        B = x.shape[0]
        
        if self.trainer.mode == 'low_rank':
            mu, D, V = params
            # Sigma = D + V @ V^T
            # Compute low rank part: V @ V^T
            Sigma = torch.bmm(V, V.transpose(1, 2))
            
            # Add diagonal D efficiently
            # We create an index for the diagonal to avoid creating a full diagonal matrix first
            diag_indices = torch.arange(self.y_dim, device=x.device)
            Sigma[:, diag_indices, diag_indices] += D
            
        else:
            mu, L = params
            # Sigma = L @ L^T
            Sigma = torch.bmm(L, L.transpose(1, 2))
            
        return mu, Sigma

    def get_projection_distribution(self, x, projection_matrix):
        """
        Computes the distribution of (projection_matrix @ Y) | X.
        
        Args:
            x: Input tensor (B, input_dim)
            projection_matrix: (K, y_dim) or (B, K, y_dim)
            
        Returns:
            mu_proj: (B, K)
            Sigma_proj: (B, K, K) - Full covariance matrix of the projection
        """
        B = x.shape[0]
        params = self.forward(x)
        
        # Handle Projection Matrix Shape
        if projection_matrix.dim() == 2:
            P = projection_matrix.unsqueeze(0).expand(B, -1, -1)  # (B, K, Y)
        else:
            P = projection_matrix
            
        # Compute projection Mean: P @ mu
        if self.trainer.mode == 'low_rank':
            mu, D, V = params
        else:
            mu, L = params
            
        mu_proj = torch.bmm(P, mu.unsqueeze(-1)).squeeze(-1) # (B, K)

        # Compute projection Covariance: P @ Sigma @ P^T
        # We construct Sigma_proj directly without instantiating full Sigma if possible
        
        if self.trainer.mode == 'low_rank':
            # Sigma = D + V V^T
            # Sigma_proj = (P @ diag(D) @ P^T) + (P @ V) @ (V^T @ P^T)
            
            # Part A: P @ diag(D) @ P^T
            # Equivalent to sum(P[i, k] * D[k] * P[j, k])
            # Implementation: (P * D) @ P^T
            D_vec = D  # (B, Y)
            P_scaled = P * D_vec.unsqueeze(1) # Broadcast D over rows of P
            term_diag = torch.bmm(P_scaled, P.transpose(1, 2))
            
            # Part B: (P @ V) @ (P @ V)^T
            V_proj = torch.bmm(P, V) # (B, K, Rank)
            term_low_rank = torch.bmm(V_proj, V_proj.transpose(1, 2))
            
            Sigma_proj = term_diag + term_low_rank
            
        else:
            # Sigma = L L^T
            # Sigma_proj = P (L L^T) P^T = (P L) @ (P L)^T
            L_proj = torch.bmm(P, L)
            Sigma_proj = torch.bmm(L_proj, L_proj.transpose(1, 2))
            
        return mu_proj, Sigma_proj

    def get_revealed_distribution(self, x, idx_revealed, values_revealed):
        """
        Computes the distribution of Y_hidden | Y_revealed = values_revealed.
        
        Args:
            x: Input tensor (B, input_dim)
            idx_revealed: List or 1D Tensor of indices that are revealed.
            values_revealed: Tensor (B, len(idx_revealed)) containing known values.
            
        Returns:
            mu_cond: (B, num_hidden) - Updated mean of hidden variables
            Sigma_cond: (B, num_hidden, num_hidden) - Updated covariance of hidden variables
        """
        B = x.shape[0]
        device = x.device
        y_dim = self.y_dim
        
        # Identify Hidden Indices
        all_indices = torch.arange(y_dim, device=device)
        # Create a mask for efficient indexing
        mask_revealed = torch.zeros(y_dim, dtype=torch.bool, device=device)
        mask_revealed[idx_revealed] = True
        idx_hidden = all_indices[~mask_revealed]
        
        len_r = len(idx_revealed)
        len_h = len(idx_hidden)
        
        if len_h == 0:
            # Everything revealed, no distribution left
            return torch.zeros(B, 0, device=device), torch.zeros(B, 0, 0, device=device)
            
        # Get Full Mean and Covariance (or Blocks)
        # For simplicity and robustness in conditioning, we reconstruct the relevant 
        # blocks of Sigma. Since we need to invert Sigma_11, explicit construction 
        # is often cleanest unless Y is massive.
        
        params = self.forward(x)
        
        if self.trainer.mode == 'low_rank':
            mu, D, V = params
            # Construct Full Sigma = diag(D) + V @ V^T
            # Note: For very high dimensions, we could use Woodbury for inversion, 
            # but usually we need the explicit Sigma_cond output anyway.
            
            # We need blocks: 11 (revealed, revealed), 12 (revealed, hidden), 
            #                 21 (hidden, revealed),   22 (hidden, hidden)
            
            # Extract sub-components
            mu_1 = mu[:, idx_revealed]
            mu_2 = mu[:, idx_hidden]
            
            # Build Sigma blocks efficiently
            # V1 = V[revealed], V2 = V[hidden]
            V1 = V[:, idx_revealed, :]
            V2 = V[:, idx_hidden, :]
            D1 = D[:, idx_revealed]
            D2 = D[:, idx_hidden]
            
            # Sigma_11 = diag(D1) + V1 @ V1^T
            Sigma_11 = torch.bmm(V1, V1.transpose(1, 2))
            diag_idx_1 = torch.arange(len_r, device=device)
            Sigma_11[:, diag_idx_1, diag_idx_1] += D1
            
            # Sigma_12 = V1 @ V2^T (Diagonal part is 0 because indices distinct)
            Sigma_12 = torch.bmm(V1, V2.transpose(1, 2))
            
            # Sigma_21 = Sigma_12^T
            Sigma_21 = Sigma_12.transpose(1, 2)
            
            # Sigma_22 = diag(D2) + V2 @ V2^T
            Sigma_22 = torch.bmm(V2, V2.transpose(1, 2))
            diag_idx_2 = torch.arange(len_h, device=device)
            Sigma_22[:, diag_idx_2, diag_idx_2] += D2
            
        else:
            mu, L = params
            mu_1 = mu[:, idx_revealed]
            mu_2 = mu[:, idx_hidden]
            
            # Construct full Sigma = L @ L^T
            Sigma = torch.bmm(L, L.transpose(1, 2))
            
            # Extract blocks
            # We use fancy indexing. We must expand dims for batching.
            # Sigma is (B, Y, Y)
            
            # Create meshgrids for block extraction
            # This is slightly verbose but precise
            ind_r = idx_revealed
            ind_h = idx_hidden
            
            Sigma_11 = Sigma[:, ind_r[:, None], ind_r]
            Sigma_12 = Sigma[:, ind_r[:, None], ind_h]
            Sigma_21 = Sigma[:, ind_h[:, None], ind_r]
            Sigma_22 = Sigma[:, ind_h[:, None], ind_h]

        # Compute Conditional Distribution
        # Formula: mu_2|1 = mu_2 + Sigma_21 * Sigma_11^-1 * (val_revealed - mu_1)
        #          Sigma_21|1 = Sigma_22 - Sigma_21 * Sigma_11^-1 * Sigma_12
        
        # Robust Inverse of Sigma_11 using Cholesky
        # Sigma_11 is symmetric positive definite
        
        # Add jitter for stability
        jitter = 1e-6 * torch.eye(len_r, device=device).unsqueeze(0)
        L_11 = torch.linalg.cholesky(Sigma_11 + jitter)
        
        # Calculate term: K = Sigma_21 * Sigma_11^-1
        # We solve (Sigma_11 * K^T = Sigma_21^T) -> (Sigma_11 * K^T = Sigma_12)
        # Let X = Sigma_11^-1 * Sigma_12
        # Solved via cholesky_solve: L_11 * L_11^T * X = Sigma_12
        gain_matrix = torch.cholesky_solve(Sigma_12, L_11) # (B, len_r, len_h)
        
        # Transpose to get Sigma_21 * Sigma_11^-1
        # Term: Sigma_21 @ (Sigma_11^-1 @ Sigma_12) is not quite right dimensionally if we just mult
        # Schur complement: Sigma_22 - Sigma_21 @ (Sigma_11^-1 @ Sigma_12)
        # gain_matrix represents (Sigma_11^-1 @ Sigma_12)
        
        schur_term = torch.bmm(Sigma_21, gain_matrix) # (B, len_h, len_h)
        Sigma_cond = Sigma_22 - schur_term
        
        # Calculate Mean
        diff = (values_revealed - mu_1).unsqueeze(-1) # (B, len_r, 1)
        # adjustment = Sigma_21 @ Sigma_11^-1 @ diff
        # We can reuse gain_matrix? No, gain was for Sigma_12.
        # We need weights = Sigma_11^-1 (y - mu_1)
        weights = torch.cholesky_solve(diff, L_11)
        adjustment = torch.bmm(Sigma_21, weights).squeeze(-1)
        
        mu_cond = mu_2 + adjustment
        
        return mu_cond, Sigma_cond

    def conformalize(self, x, y, alpha):
        n = x.shape[0]
        p = int(np.ceil((n + 1) * (1 - alpha)))
        if p > n:
            raise ValueError(
                "The number of calibration samples is too low to reach the desired alpha level."
            )

        scores = self.get_standardized_score(x, y)
        scores = torch.sort(scores, descending=False).values
        self.q_alpha = scores[p].item()  

        return self.q_alpha
    
    def conformalize_missing(self, x, y, alpha):
        n = x.shape[0]
        p = int(np.ceil((n + 1) * (1 - alpha)))
        if p > n:
            raise ValueError(
                "The number of calibration samples is too low to reach the desired alpha level."
            )

        scores = self.get_standardized_missing_score(x, y)
        scores = torch.sort(scores, descending=False).values
        self.q_alpha_missing = scores[p].item()  

        return self.q_alpha_missing
    
    def conformalize_projection(self, projection_matrix, x, y, alpha):
        n = x.shape[0]
        p = int(np.ceil((n + 1) * (1 - alpha)))
        if p > n:
            raise ValueError(
                "The number of calibration samples is too low to reach the desired alpha level."
            )

        scores = self.get_projection_standardized_score(projection_matrix, x, y)
        scores = torch.sort(scores, descending=False).values
        self.q_alpha_projection = scores[p].item()  

        self.projection_matrix = projection_matrix
        return self.q_alpha_projection
    
    def conformalize_revealed(self, idx_revealed, x, y, alpha):
        n = x.shape[0]
        p = int(np.ceil((n + 1) * (1 - alpha)))
        if p > n:
            raise ValueError(
                "The number of calibration samples is too low to reach the desired alpha level."
            )

        scores = self.get_revealed_standardized_score(idx_revealed, x, y)
        scores = torch.sort(scores, descending=False).values
        self.q_alpha_revealed = scores[p].item()  

        self.idx_revealed = idx_revealed
        return self.q_alpha_revealed
    
    def get_volume(self, x):
        _, Sigma = self.get_distribution(x)
        volumes = get_volumes(Sigma, self.q_alpha)
        scaled_volumes = volumes**(1/Sigma.shape[-1])
        return scaled_volumes
    
    def get_volume_projection(self, x):
        _, Sigma = self.get_projection_distribution(x, self.projection_matrix)
        volumes = get_volumes(Sigma, self.q_alpha_projection)
        scaled_volumes = volumes**(1/Sigma.shape[-1])
        return scaled_volumes
    
    def get_volume_revealed(self, x, values_revealed):
        _, Sigma = self.get_revealed_distribution(x, self.idx_revealed, values_revealed)
        volumes = get_volumes(Sigma, self.q_alpha_revealed) 
        scaled_volumes = volumes**(1/Sigma.shape[-1])
        return scaled_volumes
    
    def get_average_volume(self, x):
        volumes = self.get_volume(x)
        return torch.mean(volumes).item()
    
    def get_average_volume_projection(self, x):
        volumes = self.get_volume_projection(x)
        return torch.mean(volumes).item()
    
    def get_average_volume_revealed(self, x, values_revealed):
        volumes = self.get_volume_revealed(x, values_revealed)
        return torch.mean(volumes).item()

    def get_cover(self, x, y):
        scores = self.get_standardized_score(x, y)
        cover = scores <= self.q_alpha
        return cover*1.0
    
    def get_cover_missing(self, x, y):
        scores = self.get_standardized_missing_score(x, y)
        cover = scores <= self.q_alpha_missing
        return cover*1.0
    
    def get_cover_projection(self, x, y):
        scores = self.get_projection_standardized_score(self.projection_matrix, x, y)
        cover = scores <= self.q_alpha_projection
        return cover*1.0
    
    def get_cover_revealed(self, x, y):
        scores = self.get_revealed_standardized_score(self.idx_revealed, x, y)
        cover = scores <= self.q_alpha_revealed
        return cover*1.0
    
    def get_coverage(self, x, y):
        cover = self.get_cover(x, y)
        return torch.mean(cover).item()
    
    def get_coverage_missing(self, x, y):
        cover = self.get_cover_missing(x, y)
        return torch.mean(cover).item()
    
    def get_coverage_projection(self, x, y):
        cover = self.get_cover_projection(x, y)
        return torch.mean(cover).item()
    
    def get_coverage_revealed(self, x, y):
        cover = self.get_cover_revealed(x, y)
        return torch.mean(cover).item()

