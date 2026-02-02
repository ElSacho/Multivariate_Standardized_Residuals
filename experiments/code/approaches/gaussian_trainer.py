import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import copy

# ==========================================
# 1. The Model Architecture (Backbone + Head)
# ==========================================

class SimpleTabularMLP(nn.Module):
    """
    Robust Backbone: ResNet-MLP for Tabular Data
    """
    def __init__(self, num_cont, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.first_layer = nn.Linear(num_cont, hidden_dim)
        
        # Residual Blocks
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
            x = x + block(x)  # Skip connection
        return self.norm(x)

class RobustCovarianceHead(nn.Module):
    """
    Unified Head: Switches between Full Cholesky and Low-Rank automatically.
    """
    def __init__(self, input_dim, y_dim, init_sigma=1.0):
        super().__init__()
        self.y_dim = y_dim
        # Heuristic: Use Low Rank if output dim > 10
        if y_dim > 10:
            self.mode = 'low_rank'
            rank = int(math.ceil(math.sqrt(y_dim)))
            self.fc_mu = nn.Linear(input_dim, y_dim)
            self.fc_log_diag = nn.Linear(input_dim, y_dim)
            self.fc_factors = nn.Linear(input_dim, y_dim * rank)
            self.rank = rank
        else:
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

    def loss(self, params, y_target):
        if self.mode == 'low_rank':
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
        # 1. Create Mask (1 = Observed, 0 = Missing)
        mask = (~torch.isnan(y_target)).float()
        
        # 2. Safe Target (Fill NaNs with 0 to prevent NaN propagation in gradients)
        y_safe = torch.nan_to_num(y_target, nan=0.0)

        # ==========================================
        # Strategy A: Low-Rank (Vectorized & Fast)
        # ==========================================
        if self.mode == 'low_rank':
            mu, D, V = params
            
            # Residuals (Zero out missing residuals)
            r = (y_safe - mu) * mask 
            
            # Dimensions
            B, Y, K = V.shape
            
            # Inverse Variance (Diagonal)
            # D contains standard deviations. 
            # We set inverse variance to 0 where data is missing (Infinite uncertainty)
            inv_std = 1.0 / (D + 1e-8) # Add tiny epsilon to D to avoid div by zero
            inv_var_masked = (inv_std ** 2) * mask
            
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
        
    def get_center(self, x):
        """
        Calcule la matrice de covariance Lambda prédite à partir de l'entrée x.
        """
        params = self.forward(x)

        if self.mode == 'low_rank':
            mu, D, V = params
            return mu

        else:
            mu, L = params
            return mu
    
    def get_Lambda(self, x):
        """
        Calcule la matrice de covariance Lambda prédite à partir de l'entrée x.
        """
        params = self.forward(x)

        if self.mode == 'low_rank':
            mu, D, V = params
            # D est un écart-type
            D2 = D ** 2
            Lambda = torch.diag_embed(D2) + torch.bmm(V, V.transpose(1, 2))
            return Lambda

        else:
            mu, L = params
            Lambda = torch.bmm(L, L.transpose(1, 2))
            return Lambda
        
    def get_Lambda_inv(self, x):
        """
        Calcule l'inverse de la matrice de covariance Lambda prédite à partir de x.
        """
        params = self.forward(x)

        if self.mode == 'low_rank':
            mu, D, V = params
            B, Y, K = V.shape

            # D est un écart-type
            inv_D2 = 1.0 / (D ** 2)

            # A^{-1} = diag(1 / D^2)
            A_inv = torch.diag_embed(inv_D2)

            # Woodbury: (A + V V^T)^{-1}
            W = V * inv_D2.unsqueeze(-1)   # shape B x Y x K
            M = torch.eye(K, device=x.device).unsqueeze(0) + torch.bmm(V.transpose(1, 2), W)

            L_M = torch.linalg.cholesky(M)
            M_inv = torch.cholesky_inverse(L_M)

            correction = torch.bmm(W, torch.bmm(M_inv, W.transpose(1, 2)))
            Lambda_inv = A_inv - correction
            return Lambda_inv

        else:
            mu, L = params
            # Lambda = L L^T  ->  Lambda^{-1} via Cholesky
            Lambda_inv = torch.cholesky_inverse(L)
            return Lambda_inv
        
class CenterPredictor(nn.Module):
    def __init__(self, head, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feat = self.backbone(x)
        mu = self.head.get_center(feat)
        return mu
    
class MatrixPredictor(nn.Module):
    def __init__(self, head, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def __call__(self, x):
        if self.head.mode == "low_rank":
            feat = self.backbone(x)
            mu, D, V = self.head(feat)
            B, Y, K = V.shape

            # D is std dev -> diagonal covariance D^2
            inv_D2 = 1.0 / (D ** 2)

            # A^{-1} = diag(1 / D^2)
            A_inv = torch.diag_embed(inv_D2)

            # Woodbury components
            W = V * inv_D2.unsqueeze(-1)   # B x Y x K
            M = torch.eye(K, device=V.device).unsqueeze(0) + torch.bmm(V.transpose(1, 2), W)

            L_M = torch.linalg.cholesky(M)
            M_inv = torch.cholesky_inverse(L_M)

            correction = torch.bmm(W, torch.bmm(M_inv, W.transpose(1, 2)))
            Sigma_inv = A_inv - correction   # (D^2 I + V V^T)^{-1}

            # -------- inverse square root --------
            eigvals, eigvecs = torch.linalg.eigh(Sigma_inv)

            eps = 1e-6
            inv_sqrt_eigvals = torch.diag_embed(torch.sqrt(eigvals + eps))

            Sigma_inv_sqrt = eigvecs @ inv_sqrt_eigvals @ eigvecs.transpose(1, 2)

            return Sigma_inv_sqrt

        feat = self.backbone(x)
        mu, L = self.head(feat)

        Sigma = torch.bmm(L, L.transpose(1, 2))

        # Eigen-decomposition
        eigvals, eigvecs = torch.linalg.eigh(Sigma)

        # Numerical safety
        eps = 1e-6
        inv_sqrt_eigvals = torch.diag_embed(1.0 / torch.sqrt(eigvals + eps))

        Sigma_inv_sqrt = eigvecs @ inv_sqrt_eigvals @ eigvecs.transpose(1, 2)
        return Sigma_inv_sqrt
    
    def get_Sigma(self, x):
        feat = self.backbone(x)
        Sigma = self.head.get_Lambda_inv(feat)
        return Sigma
    

class GaussianTrainer:
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=0.1, init_sigma=1.0):
        self.backbone = SimpleTabularMLP(num_cont=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.head = RobustCovarianceHead(input_dim=hidden_dim, y_dim=output_dim, init_sigma=init_sigma)
    
    def get_Lambdas(self, x):
        """
        Compute the matrix Lambda = Sigma^{-1/2} for a given input x.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (n, d)
        
        Returns
        -------
        Lambdas : torch.Tensor
            The matrix Lambda(X) = Sigma(X)^{-1/2} of shape (n, k, k) such that Y = N( f(x), Sigma(X) )
        """
        Lambdas = self.matrix_model(x)
        return Lambdas
    
    def get_centers(self, x):
        """
        Compute the centers of the ellipsoids for a given input x.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (n, d)
        
        Returns
        -------
        centers : torch.Tensor
            The centers of the ellipsoids of shape (n, k)
        """
        return self.center_model(x)
    
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

                feats = self.backbone(bx)
                preds = self.head(feats)

                if not NaN_Values: loss = self.head.loss(preds, by)
                else : loss = self.head.loss_with_missing(preds, by)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(trainloader)
        
            self.backbone.eval()
            self.head.eval()
            total_stop_loss = 0.0

            with torch.no_grad():
                for bx, by in stoploader:
            
                    feats = self.backbone(bx)
                    preds = self.head(feats)

                    if not NaN_Values: loss = self.head.loss(preds, by)
                    else : loss = self.head.loss_with_missing(preds, by)
                    
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

        self.center_model = CenterPredictor(self.head, self.backbone)
        self.matrix_model = MatrixPredictor(self.head, self.backbone)