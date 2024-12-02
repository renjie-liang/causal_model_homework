import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads=8, hidden_size=256, num_layers=4, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Linear(input_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.regressor = nn.Linear(hidden_size // 4, 1)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.embedding(x)  # shape: (batch_size, seq_len, hidden_size)

        # Permute for transformer input
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)

        # Transformer expects input of shape (seq_len, batch_size, hidden_size)
        transformer_out = self.transformer_encoder(x)  # (seq_len, batch_size, hidden_size)

        # Get the last timestep output
        hidden = transformer_out[-1, :, :]  # (batch_size, hidden_size)
        
        out = F.relu(self.fc1(hidden))
        out = self.dropout_layer(out)
        out = F.relu(self.fc2(out))
        out = self.dropout_layer(out)
        out = self.regressor(out)
        return out.squeeze(1)  # Return shape should match target shape (batch_size,)