import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


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

        output = output * self.upper_triangular_mask

        return output
    
    def init_weights(self, m, std = 0.02):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)  # N(0, 0.02^2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



class MatrixPredictor(nn.Module):
    def __init__(self, input_dim, output_rows, output_cols, hidden_dim=100, n_hidden_layers=0, dropout_rate=0.0):
        super(MatrixPredictor, self).__init__()
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

        return output
    
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
    
    def fit_and_plot(self, train_loader, validation_loader, epochs, keep_best = True, lr = 0.001, verbose=False):
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
                loss = F.mse_loss(output, Y_batch)
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
                print(f'Epoch: {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {validation_loss}')
        if keep_best:
            self.load_state_dict(best_weights)
        return train_losses, validation_losses

    def freeze_all_but_last_layer(self):
        for name, param in self.named_parameters():
            if 'fc3' not in name:
                param.requires_grad = False

    def eval(self, loader):
        with torch.no_grad():
            loss = 0
            for i, (X_batch, Y_batch) in enumerate(loader):
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
