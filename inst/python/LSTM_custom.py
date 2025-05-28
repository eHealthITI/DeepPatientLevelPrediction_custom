import torch
import math
from torch import nn
from ResNet import NumericalEmbedding

class LSTM_custom(nn.Module):
    def __init__(
            self, 
            cat_features,
            num_features=0,
            hidden_size=128, 
            num_layers=3, 
            dropout=0.2, 
            bidirectional=True, 
            size_embedding=256,  # Size of the embedding for cat_features and num_features
            dim_out=1,       # Output dimension set to 1
            model_type="LSTM_custom"
        ):
        super(LSTM_custom, self).__init__()

        self.name = model_type
        cat_features = int(cat_features)
        num_features = int(num_features)
        size_embedding = int(size_embedding)
        hidden_size = int(hidden_size)

        # Store bidirectional and hidden size for later use
        self.bidirectional = bidirectional
        self.size_hidden = hidden_size

        # Embedding layers for categorical and numerical features
        self.cat_embedding = nn.EmbeddingBag(
            cat_features + 1, size_embedding, padding_idx=0
        )
        
        if num_features != 0 and num_features is not None:
            self.num_embedding = NumericalEmbedding(num_features, size_embedding)

        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=size_embedding * 2,  # Input size includes both cat and num features
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Define a fully connected output layer with dim_out=1
        direction_multiplier = 2 if bidirectional else 1
        self.output = nn.Linear(hidden_size * direction_multiplier, dim_out)  # dim_out=1

        # Additional layers for normalization and activation
        self.last_norm = nn.BatchNorm1d(hidden_size * direction_multiplier)
        self.last_act = nn.ReLU()

    def forward(self, input):
        # Retrieve categorical and numerical data from input
        x_cat = input["cat"]
        x_cat_embed = self.cat_embedding(x_cat)  # Categorical embedding
        
        if "num" in input.keys() and self.num_embedding is not None:
            x_num = input["num"]
            x_num_embed = self.num_embedding(x_num)  # Numerical embedding
            # Combine embeddings (taking mean for numerical features, similar to MLP)
            x = torch.cat([x_cat_embed, x_num_embed.mean(dim=1)], dim=1)  # [batch_size, 512]

        else:
            x = x_cat_embed

        # LSTM expects a 3D tensor [batch_size, seq_length, input_size]
        # Assuming we want to feed the embeddings as the sequence input
        x_embed = x.unsqueeze(1)  # Add a sequence dimension (seq_length=1 for now)

        # Pass through the LSTM
        out, _ = self.lstm(x_embed)

        # Take the last output (sequence output)
        last_hidden = out[:, -1, :]

        # Normalize and apply activation
        x = self.last_norm(last_hidden)
        x = self.last_act(x)

        # Pass through the output layer (dim_out=1)
        x = self.output(x)

        return x.squeeze(-1)  # Remove the extra dimension

    def reset_head(self):
        # This method might not be necessary, but if you still need it:
        direction_multiplier = 2 if self.bidirectional else 1
        self.output = nn.Linear(self.size_hidden * direction_multiplier, 1)
