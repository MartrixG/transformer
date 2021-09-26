import torch
from torch import nn
from self_attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, word_em_dim, heads):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(word_em_dim, heads=heads)
        self.norm1 = nn.LayerNorm(word_em_dim)
        self.norm2 = nn.LayerNorm(word_em_dim)
        self.ff = nn.Sequential(
            nn.Linear(word_em_dim, 4 * word_em_dim),
            nn.ReLU(),
            nn.Linear(4 * word_em_dim, word_em_dim)
        )

    def forward(self, x):
        attend = self.attention(x)
        x = self.norm1(attend + x)
        res = self.ff(x)
        return self.norm2(res + x)


if __name__ == '__main__':
    x_in = torch.rand((32, 128, 300))
    network = TransformerBlock(300, 8)
    print(x_in)
    y = network(x_in)
    print(y)
