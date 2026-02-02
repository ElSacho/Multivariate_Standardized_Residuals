import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

class PSDMatrixPredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=100,
        n_hidden_layers=0,
        dropout_rate=0.0,
    ):
        super().__init__()

        self.output_dim = output_dim

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.tab_hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        )

        # Number of parameters
        self.n_scales = output_dim
        self.n_corr = output_dim * (output_dim - 1) // 2

        self.fc_out = nn.Linear(hidden_dim, self.n_scales + self.n_corr)

        # Mask for strictly lower triangular part
        self.register_buffer(
            "lower_triangular_mask",
            torch.tril(torch.ones(output_dim, output_dim), diagonal=-1),
        )

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.dropout(x)

        for layer in self.tab_hidden:
            x = F.relu(layer(x))
            x = self.dropout(x)

        params = self.fc_out(x)

        # --- split parameters ---
        log_scales = params[:, : self.n_scales]
        corr_params = params[:, self.n_scales :]

        # --- diagonal scale matrix D ---
        scales = torch.exp(log_scales)
        D = torch.diag_embed(scales)

        # --- build Cholesky of correlation ---
        batch_size = x.shape[0]
        L_R = torch.zeros(batch_size, self.output_dim, self.output_dim, device=x.device)

        # unit diagonal
        L_R[:, torch.arange(self.output_dim), torch.arange(self.output_dim)] = 1.0

        # fill strictly lower triangular part
        idx = torch.tril_indices(self.output_dim, self.output_dim, offset=-1)
        L_R[:, idx[0], idx[1]] = torch.tanh(corr_params)

        # --- correlation matrix ---
        R = L_R @ L_R.transpose(-1, -2)

        # --- final PSD matrix ---
        Sigma = D @ R @ D

        return Sigma
    
    def get_dets(self, x):
        x = F.relu(self.fc(x))
        x = self.dropout(x)

        for layer in self.tab_hidden:
            x = F.relu(layer(x))
            x = self.dropout(x)

        params = self.fc_out(x)

        log_scales = params[:, : self.n_scales]

        # det(Sigma) = prod_i exp(2 * log_scale_i)
        det = torch.exp(2.0 * torch.sum(log_scales, dim=1))

        return det



class CholeskyMatrixPredictor(nn.Module):
    def __init__(self, input_dim, output_rows, output_cols, hidden_dim=100, n_hidden_layers=0, dropout_rate=0.0):
        super(CholeskyMatrixPredictor, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.n_hidden_layers = n_hidden_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.tab_hidden = nn.ModuleList()
        for _ in range(n_hidden_layers):
            self.tab_hidden.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc2 = nn.Linear(hidden_dim, output_rows * output_cols)
        self.output_rows = output_rows
        self.output_cols = output_cols
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        # Triangular mask to ensure the output is upper triangular
        self.register_buffer("upper_triangular_mask", torch.triu(torch.ones(output_rows, output_cols)))  
    
    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        for hidden in self.tab_hidden:
            x = F.relu(hidden(x))
            x = self.dropout(x)
        output = self.fc2(x)
        output = output.view(-1, self.output_rows, self.output_cols)

        output = output * self.upper_triangular_mask

        Lambdas = torch.einsum('nij,nkj->nik', output, output)

        return Lambdas


class MatrixPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100, n_hidden_layers=0, dropout_rate=0.0):
        super(MatrixPredictor, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.n_hidden_layers = n_hidden_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.tab_hidden = nn.ModuleList()
        for _ in range(n_hidden_layers):
            self.tab_hidden.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc2 = nn.Linear(hidden_dim, output_dim * output_dim)
        self.output_rows = output_dim
        self.output_cols = output_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

    def re_init_weights(self):
        """
        Réinitialise les poids du modèle avec une loi normale N(0, std^2).
        """
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.tab_hidden = nn.ModuleList()
        for _ in range(self.n_hidden_layers):
            self.tab_hidden.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc2 = nn.Linear(self.hidden_dim, self.output_rows * self.output_cols)
        
    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        for hidden in self.tab_hidden:
            x = F.relu(hidden(x))
            x = self.dropout(x)
        output = self.fc2(x)
        output = output.view(-1, self.output_rows, self.output_cols)

        Lambdas = torch.einsum('nij,nkj->nik', output, output)
        return Lambdas
    
    def init_weights(self, m, std = 0.02):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)  # Loi normale N(0, 0.02^2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Network(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dim=100, n_hidden_layers=1, dropout_rate=0.0):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.tab_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.fc3 = nn.Linear(hidden_dim, output_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        for hidden in self.tab_hidden:
            x = F.relu(hidden(x))
            x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def fit(self, train_loader, validation_loader, epochs, keep_best=True, lr=0.001, verbose=False, device="cpu"):
        if not keep_best:
            print("Training... (keep the last model)")

        self.to(device)

        best_weights = None
        best_loss = float("inf")

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_losses = []
        validation_losses = []

        for epoch in range(epochs):
            self.train()
            for X_batch, Y_batch in train_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                optimizer.zero_grad()
                output = self(X_batch)
                loss = F.mse_loss(output, Y_batch)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                train_loss = self.eval_loss(train_loader, device)
                validation_loss = self.eval_loss(validation_loader, device)

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

            if keep_best and validation_loss < best_loss:
                best_loss = validation_loss
                best_weights = copy.deepcopy(self.state_dict())

            if verbose:
                print(f"Epoch: {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {validation_loss}")

        if keep_best and best_weights is not None:
            self.load_state_dict(best_weights)

        return train_losses, validation_losses


    def freeze_all_but_last_layer(self):
        for name, param in self.named_parameters():
            if 'fc3' not in name:
                param.requires_grad = False

    def eval_loss(self, loader, device):
        with torch.no_grad():
            loss = 0
            for i, (X_batch, Y_batch) in enumerate(loader):
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                output = self(X_batch)
                loss += F.mse_loss(output, Y_batch)
            return loss / len(loader)
        
class LinearNetwork(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LinearNetwork, self).__init__()
        self.fc1 = nn.Linear(input_channels, output_channels)


    def forward(self, x):
        x = self.fc1(x)
        return x

    def fit(self, train_loader, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            for i, (X_batch, Y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(X_batch)
                loss = F.mse_loss(output, Y_batch)
                loss.backward()
                optimizer.step()
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')


    def eval(self, X_val, Y_val):
        with torch.no_grad():
            output = self(X_val)
            loss = F.mse_loss(output, Y_val)
            print(f'Validation Loss: {loss.item()}')

class NetworkWithMissingOutputs(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dim = 100, n_hidden_layers = 1):
        super(NetworkWithMissingOutputs, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.tab_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.fc3 = nn.Linear(hidden_dim, output_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for hidden in self.tab_hidden:
            x = F.relu(hidden(x))
        x = self.fc3(x)
        return x

    def fit(self, train_loader, epochs, verbose=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            for i, (X_batch, Y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(X_batch)
                loss = F.mse_loss(output, Y_batch)
                loss.backward()
                optimizer.step()
            if verbose:
                print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
    
    def fit_and_plot(self, train_loader, validation_loader, epochs, keep_best = False, lr = 0.001, verbose=False):
        if not keep_best:
            print("Training... (keep the last model)")
        best_weights = None
        best_loss = float('inf')
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_losses = []
        validation_losses = []
        for epoch in range(epochs):
            for i, (X_batch, Y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(X_batch)
                mask = ~torch.isnan(Y_batch)
                if mask.any():
                    loss = F.mse_loss(output[mask], Y_batch[mask])
                    loss.backward()
                    optimizer.step()
            train_loss = self.eval(train_loader)
            validation_loss = self.eval(validation_loader)
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            if keep_best and validation_loss < best_loss:
                best_loss = validation_loss
                best_weights = copy.deepcopy(self.state_dict())
            if verbose:
                print(f'Epoch: {epoch + 1}, Train Loss: {train_loss}, validation Loss: {validation_loss}')
        if keep_best:
            self.load_state_dict(best_weights)
        return train_losses, validation_losses

    def freeze_all_but_last_layer(self):
        for name, param in self.named_parameters():
            if 'fc3' not in name:
                param.requires_grad = False

    def eval(self, loader):
        with torch.no_grad():
            total_loss = 0.0
            total_count = 0  

            for i, (X_batch, Y_batch) in enumerate(loader):
                output = self(X_batch)

                mask = ~torch.isnan(output) & ~torch.isnan(Y_batch)

                valid_outputs = output[mask]
                valid_targets = Y_batch[mask]

                if valid_outputs.numel() > 0:
                    total_loss += F.mse_loss(valid_outputs, valid_targets, reduction='sum').item()
                    total_count += valid_outputs.numel()
                else:
                    print(f"Batch {i} ignoré : aucun élément valide (tout est NaN)")

            return total_loss / total_count if total_count > 0 else float('nan')


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
    