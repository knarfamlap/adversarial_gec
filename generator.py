import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb


class Generator(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 vocab_size,
                 max_seq_len,
                 device,
                 oracle_init=False):
        super(Generator, self).__init__()
        # hidden state dim for GRU
        self.hidden_dim = hidden_dim
        # embedding dim for embedding layer
        self.embedding_dim = embedding_dim
        # maximum length of a sequence
        self.max_seq_len = max_seq_len
        # vocaubalary size
        self.vocab_size = vocab_size
        # cuda or cpu
        self.device = device
        # initialize an embedding for every word on the vocab
        # embedding vectors are size of embedding dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # init the GRU
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        # linear layer that gives the logits
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

        if oracle_init:
            for p in self.parameters():
                nn.init.normal_(p)

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)

        return h

    def forward(self, x, hidden):
        """
        Passes input into embedding and applies GRU per token per time step
        """
        # each row represents a seq
        emb = self.embeddings(x)  # batch_sz * embedding_dim
        # add an extra dimension
        emb = emb.view(1, -1,
                       self.embedding_dim)  # 1 * batch_sz * embedding_dim
        # get output and hidden layer
        out, hidden = self.gru(emb, hidden)  # 1 * batch_sz * hiddin_dim
        # flatten the dimension (remove extra dim that was put in)
        out = self.gru2out(out.view(-1,
                                    self.hidden_dim))  # batch_sz * vocab_sz
        # pass into log softmax. Convert them into probabilities
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, n, start_letter=0):
        """
        Samples networks and returns n samples of length max_seq_len

        """
        # return zero matrix with n rows and max_seq_len rows
        samples = torch.zeros(n, self.max_seq_len, device=self.device)
        # init hidden state
        h = self.init_hidden(n)
        # create a long tensor
        x = torch.tensor([start_letter] * n,
                         dtype=torch.long, device=self.device)

        for i in range(self.max_seq_len):
            out, h = self.forward(x, h)  # out: (n, vocab_size)
            out = torch.multinomial(torch.exp(out), 1)  # (n x 1)
            samples[:, i] = out.view(-1).data

            x = out.view(-1)

        return samples

    def batchNLLLoss(self, inp, target):
        """
        Get the NLLLoss per batch
        """
        # inint the loss function
        loss_fn = nn.NLLLoss()
        
        batch_size, seq_len = inp.size()
        # reverse dimensions
        inp = inp.permute(1, 0)  # seq_len * batch_sz
        # reverse dimensions
        target = target.permute(1, 0)
        # init hidden state with 
        h = self.init_hidden(batch_size)

        loss = 0

        for i in range(seq_len):
            # get the out and hidden state 
            out, h = self.forward(inp[i], h)
            # calculates the loss at every timestep
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
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)
        target = target.permute(1, 0)
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)

            for j in range(batch_size):
                loss += -out[j][target.data[i][j]] * reward[j]

        return loss / batch_sz
