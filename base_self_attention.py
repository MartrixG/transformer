import torch
import torch.nn.functional as F

b, t, k = 2, 3, 5

if __name__ == '__main__':
    # init input tokens of sentence, x shape is (b, t, k)
    x = torch.rand((b, t, k), dtype=torch.float32)
    print(x)

    # init attention, w shape is (b, t, t)
    weights_base = torch.bmm(x, x.transpose(1, 2))
    weights = F.softmax(weights_base, dim=2)
    print(weights)

    # calc new x after attention, y shape is (b, t, k)
    y = torch.bmm(weights, x)
    print(y)
