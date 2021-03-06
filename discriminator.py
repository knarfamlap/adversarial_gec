import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_sz: int, max_seq_len: int, device: str, dropout=0.3):
        super(Discriminator, self).__init__()
        # hidden dimension for the hidden state
        self.hidden_dim = hidden_dim
        # embedding dimensions for size of embedding vector
        self.embedding_dim = embedding_dim
        # maximum sequence length
        self.max_seq_len = max_seq_len
        self.device = device

        self.embeddings = nn.Embedding(vocab_sz, embedding_dim).to(device)
        self.gru = nn.GRU(embedding_dim, hidden_dim,
                          num_layers=2, bidirectional=True, dropout=dropout).to(device)
        self.gru2hidden = nn.Linear(2 * 2 * hidden_dim, hidden_dim).to(device)
        self.dropout_linear = nn.Dropout(p=dropout).to(device)
        self.hidden2out = nn.Linear(hidden_dim, 1).to(device)

    def init_hidden(self, batch_sz: int) -> torch.Tensor:
        h = torch.zeros(2*2*1, batch_sz, self.hidden_dim, device=self.device)

        return h

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # embed input to batch_sz * seq_len * embedding_dim
        embed = self.embeddings(x)
        # change dimentions
        embed = embed.permute(1, 0, 2)  # seq_len * batch_sz * embedding_dim
        _, hidden = self.gru(embed, hidden)  # 4 * batch_sz * hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()
        out = self.gru2hidden(hidden.view(-1, 4 * self.hidden_dim))
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)
        out = torch.sigmoid(out)
        return out

    def batchClassify(self, inp: torch.Tensor) -> torch.Tensor:
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h)
        return out.view(-1)

    def batchBCELoss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_fn = nn.BCELoss()
        h = self.init_hidden(x.size()[0])
        out = self.forward(x, h)

        return loss_fn(out, target)
