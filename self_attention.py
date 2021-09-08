import torch
import torch.nn.functional as F

from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super(SelfAttention, self).__init__()
        self.dim, self.heads = dim, heads
        self.Q = nn.Linear(dim, dim * heads, bias=False)
        self.K = nn.Linear(dim, dim * heads, bias=False)
        self.V = nn.Linear(dim, dim * heads, bias=False)
        self.unify = nn.Linear(dim * heads, dim, bias=False)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        Q = self.Q(x).reshape(b, t, k, h)
        K = self.K(x).reshape(b, t, k, h)
        V = self.V(x).reshape(b, t, k, h)

        Q = Q.transpose(1, 3).reshape(b * h, t, k)
        K = K.transpose(1, 3).reshape(b * h, t, k)
        V = V.transpose(1, 3).reshape(b * h, t, k)

        Q /= (k ** (1/4))
        K /= (k ** (1/4))

        dot = torch.bmm(Q, K.transpose(1, 2))
        dot = torch.softmax(dot, dim=2)

        out = torch.bmm(dot, V).reshape(b, h, t, k)
        out = out.transpose(1, 2).reshape(b, t, h * k)

        return self.unify(out)


if __name__ == '__main__':
    net = SelfAttention(5)
    x_in = torch.rand((2, 3, 5))
    print(x_in)
    print(net(x_in))
    # init input tokens of sentence, x shape is (b, t, k)
    # init q, k, v matrix with shape of (k, k)
