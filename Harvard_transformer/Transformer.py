import torch

from model import make_model
from utils import LabelSmoothing, NoamOpt, data_gen, run_epoch, greedy_decode, SimpleLossCompute, batch_size_fn
from torchtext.legacy import data, datasets
from utils import Batch
from torch import nn


def rand_data_test():
    vocab_size = 11
    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.0)
    network = make_model(vocab_size, vocab_size, N=2)
    model_opt = NoamOpt(network.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(network.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(5):
        network.train()
        run_epoch(data_gen(vocab_size, 30, 20), network, SimpleLossCompute(network.generator, criterion, model_opt))
        network.eval()
        print(run_epoch(data_gen(vocab_size, 30, 5), network, SimpleLossCompute(network.generator, criterion, None)))

    network.eval()
    input_tokens = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    input_mask = torch.ones((1, 1, 10))
    print(greedy_decode(network, input_tokens, input_mask, max_len=10, start_symbol=1))


def train():
    import spacy
    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)
    MAX_LEN = 100
    # 数据集无法下载。。。。
    train_data, val_data, test_data = \
        datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT),
                              filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train_data.trg, min_freq=MIN_FREQ)

    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train_data, batch_size=BATCH_SIZE, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val_data, batch_size=BATCH_SIZE, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

    for epoch in range(10):
        model.train()
        run_epoch((get_batch(pad_idx, b) for b in train_iter),
                  model,
                  SimpleLossCompute(model.generator, criterion, opt=model_opt))
        model.eval()
        loss = run_epoch((get_batch(pad_idx, b) for b in valid_iter),
                         model,
                         SimpleLossCompute(model.generator, criterion, opt=None))
        print(loss)


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def get_batch(pad_idx, batch):
    """
    Fix order in torchtext to match ours
    """
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


if __name__ == '__main__':
    train()
