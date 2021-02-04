import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb


class Generator(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 vocab_sz,
                 max_seq_len,
                 gpu=False,
                 oracle_init=False):
        super(Generator, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_sz = vocab_sz
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_sz, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_sz)

        if oracle_init:
            for p in self.parameters():
                nn.init.normal_(p)

    def init_hidden(self, batch_sz=1):
        h = torch.zeros(1, batch_sz, self.hidden_dim, requires_grad=True)

        if self.gpu:
            return h.cuda()

        return h

    def forward(self, x, hidden):
        """
        Passes input into embedding and applies GRU per token per time step
        """
        emb = self.embeddings(x)  # batch_sz * embedding_dim
        emb = emb.view(1, -1,
                       self.embedding_dim)  # 1 * batch_sz * embedding_dim
        out, hidden = self.gru(emb, hidden)  # 1 * batch_sz * hiddin_dim
        out = self.gru2out(out.view(-1,
                                    self.hidden_dim))  # batch_sz * vocab_sz
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, n, start_letter=0):
        """
        Samples networks and returns n samples of length max_seq_len

        """
        samples = torch.zeros(n, self.max_seq_len).type(torch.LongTensor)
        h = self.init_hidden(n)
        x = torch.LongTensor([start_letter] * n, requires_grad=True)

        if self.gpu:
            samples = samples.cuda()
            x = x.cuda()

        for i in range(self.max_seq_len):
            out, h = self.forward(x, h)  # out: n * vocab_sz
            out = torch.multinomial(torch.exp(out), 1)  # n * 1
            samples[:, 1] = out.view(-1).data

            x = out.view(-1)

        return samples

    def batchNLLLoss(self, inp, target):

        loss_fn = nn.NLLLoss()
        batch_sz, seq_len = inp.size()
        inp = inp.permute(1, 0)  # seq_len * batch_sz
        target = inp.permute(1, 0)
        h = self.init_hidden(batch_sz)

        loss = 0

        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += loss_fn(out, target[i])

        return loss  # per batch

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/
        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)
            inp should be target with <s> (start letter) prepended
        """
        batch_sz, seq_len = inp.size()
        inp = inp.permute(1, 0)
        target = target.permute(1, 0)
        h = self.init_hidden(batch_sz)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)

            for j in range(batch_sz):
                loss += -out[j][target.data[i][j]] * reward[j]

        return loss / batch_sz
