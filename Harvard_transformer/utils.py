import time

import numpy as np
import torch.nn as nn
import torch


class Batch:
    """
    Object for holding a batch of data with mask during training.
    """

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            # in the decoder, we need predict next word base on the what we have generate.
            # trg remove the last word and trg_y remove the first word, so when decoder generate trg[i],
            # the next word is expected as trg_y[i]
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.n_tokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        tgt的形状是（batch_size, token_length）而在后续的处理中，经过词嵌入以后形状会变为（batch_size, token_length, d_model）
        在计算multi-head attention的情况下，会变成四维向量：（batch_size, h, token_length, d_model）
        由于不同的head的mask是相同的，所以只需要在第一个维度添加一个占位维度即可。
        所以在生成tgt_mask时，需要在（batch_size, token_length）的中间加一个占位维度。
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        res = subsequent_mask(tgt.size(-1))
        tgt_mask = tgt_mask & res
        return tgt_mask


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    # make the num matrix to a bool matrix
    return np.triu(np.ones((1, size, size), dtype=np.int8), k=1) == 0


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count):
    """
    Keep augmenting batch and calculate total number of tokens + padding.
    """
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    """
    Optimizer wrapper that implements rate.
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """
        Update parameters and rate
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        Implement `l_rate` above"
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model_size=model.src_embed[0].d_model,
                   factor=2,
                   warmup=4000,
                   optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def data_gen(V, batch, batches):
    """
    Generate random data for a src-tgt copy task.
    """
    for i in range(batches):
        data = np.random.randint(1, V, size=(batch, 10), dtype=np.int64)
        data[:, 0] = 1
        src = torch.from_numpy(data)
        tgt = torch.from_numpy(data)
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    """
    A simple loss compute and train function.
    """

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)

        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm


class LabelSmoothing(nn.Module):
    """
    Implement label smoothing.
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


def run_epoch(data_iter, model, loss_compute):
    """
    Standard Training and Logging Function
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.n_tokens)
        total_loss += loss
        total_tokens += batch.n_tokens
        tokens += batch.n_tokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.n_tokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
