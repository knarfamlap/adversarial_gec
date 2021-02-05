import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_sz, max_seq_len, gpu=False, dropout=0.3):
        super(Discriminator, self).__init__() 
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu


        self.embeddings = nn.Embedding(vocab_sz, embedding_dim) 
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout) 
        self.gru2hidden = nn.Linear(2 * 2 * hidden_dim, hidden_dim) 
        self.dropout_linear = nn.Dropout(p=dropout) 
        self.hidden2out = nn.Linear(hidden_dim, 1) 


    def init_hidden(self, batch_sz): 
        h = torch.zeros(2*2*1, batch_sz, self.hidden_dim, requires_grad=True) 

        if self.gpu:
            return h.cuda() 

        return h 


    def forward(self, x, hidden):
        embed = self.embeddings(x)  # embed input to batch_sz * seq_len * embedding_dim
        embed = emb.permute(1, 0, 2)  # seq_len * batch_sz * embedding_dim
        _, hidden = self.gru(emb, hidden) # 4 * batch_sz * hidden_dim
        hidden = hidden.permute(1, 0, 2).contigous() 
        out = self.gru2hidden(hidden.view(-1, 4  * self.hidden_dim))
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)
        out = torch.sigmoid(out)
        return out



    def batchClassify(self, inp):
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h) 
        return out.view(-1) 


    def batchBCELoss(self, x, target):
        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0]) 
        out = self.forward(inp, h) 

        return loss_fn(out, target) 

