import torch
import torch.nn as nn

class SimpleMLP(nn.Module):

    def __init__(self, dim_in, dim_inner=128, dropout_rate=0.9):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_inner)
        self.fc2 = nn.Linear(dim_inner, dim_inner)
        self.fc3 = nn.Linear(dim_inner, 1)

        # Activations are separated for shap explainability: https://github.com/slundberg/shap/issues/1678
        self.activation_h1 = nn.Tanh()
        self.activation_h2 = nn.Tanh()
        self.activation_final = nn.Sigmoid()
        
        self.dropout_rate: float = dropout_rate
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        x = self.activation_h1(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.activation_h2(self.fc2(x))
        x = self.dropout_layer(x)
        x = self.fc3(x)

        return self.activation_final(x)
