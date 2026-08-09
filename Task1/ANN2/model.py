import torch
import torch.nn as nn

class NOE_GRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1, num_layers=1, use_mlp_head=False):
        super(NOE_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # The input to the GRU is the current voltage u(k) AND the past prediction y_hat(k-1)
        # For multiple layers, we stack GRUCells.
        self.gru_cells = nn.ModuleList()
        # First layer takes the physical input
        self.gru_cells.append(nn.GRUCell(input_size=(input_dim + output_dim), hidden_size=hidden_dim))
        
        # Subsequent layers take the hidden state of the previous layer
        for _ in range(1, num_layers):
            self.gru_cells.append(nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim))
        
        # Maps the internal hidden state to the physical angle prediction
        if use_mlp_head:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, u_seq, y_init, y_true_seq=None, teacher_forcing_ratio=0.0):
        """
        u_seq: Tensor of shape (Batch, Seq_Length, 1)
        y_init: Tensor of shape (Batch, 1) - The true angle right before the sequence starts
        y_true_seq: Tensor of shape (Batch, Seq_Length, 1) - True target sequence for teacher forcing
        teacher_forcing_ratio: Float [0.0, 1.0] - Probability of using true past output instead of prediction
        """
        batch_size, seq_len, _ = u_seq.size()
        device = u_seq.device
        
        y_preds = []
        y_prev = y_init # Initialize the feedback loop
        
        # Initialize GRU hidden state for each layer to zeros
        h_k = [torch.zeros(batch_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
        
        # Explicit simulation loop over the sequence chunk
        for k in range(seq_len):
            u_k = u_seq[:, k, :]
            
            # --- TEACHER FORCING ---
            if k > 0 and teacher_forcing_ratio > 0.0 and y_true_seq is not None:
                if torch.rand(1).item() < teacher_forcing_ratio:
                    y_prev = y_true_seq[:, k-1, :]
            
            # 1. NOE STRUCTURE: Concatenate u(k) and y_hat(k-1) (or y_true(k-1) if forced)
            layer_input = torch.cat([u_k, y_prev], dim=-1)
            
            # 2. Update memory state for all layers
            for layer_idx in range(self.num_layers):
                h_k[layer_idx] = self.gru_cells[layer_idx](layer_input, h_k[layer_idx])
                # The input to the next layer is the hidden state of the current layer
                layer_input = h_k[layer_idx]
            
            # 3. Predict current angle y_hat(k) from the top layer's hidden state
            y_curr = self.output_layer(h_k[-1])
            y_preds.append(y_curr)
            
            # 4. FEEDBACK LOOP: Set current prediction as previous for the next iteration
            y_prev = y_curr
            
        # Stack predictions back into (Batch, Seq_Length, 1)
        return torch.stack(y_preds, dim=1)