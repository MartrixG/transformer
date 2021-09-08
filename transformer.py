import torch
from torch import nn
from self_attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, x):
        attend = self.attention(x)
        x = self.norm1(attend + x)
        res = self.ff(x)
        return self.norm2(res + x)


if __name__ == '__main__':
    x_in = torch.rand((2, 3, 5))
    network = TransformerBlock(5, 8)
    print(x_in)
    y = network(x_in)
    print(y)
