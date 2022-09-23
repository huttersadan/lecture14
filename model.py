import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence
from typing import List
import torch.nn.functional as F
from torchtext import data

class WordAverage(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, num_cls:int, padding_idx:int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.padding_idx = padding_idx
        self.fc = nn.Linear(embedding_dim, num_cls)

    def forward(self, text_ids:torch.Tensor, length:List[int]) -> torch.Tensor:
        # TODO 4 实现输入词语id, 查表得到word embedding,
        #batch_size = text_ids.shape[0]
        embeded = self.embedding(text_ids)
        pooled = F.avg_pool2d(embeded,(embeded.shape[1],1)).squeeze(1)#batch_size*embedding_dim
        return self.fc(pooled)
        # 将句内word embedding平均得到sentence embedding,
        # 最后将sentence embedding 过全连接层进行分类的过程
        # 注意batch内各个序列的有效长度不同，平均时的分母应该不同
        pass

    def _init_embedding(self, pretrain_emb:torch.Tensor, freeze:bool=True) -> None:
        assert pretrain_emb.shape == self.embedding.weight.shape, 'invalid table shape'
        print('fuck you!')
        self.embedding = nn.Embedding.from_pretrained(pretrain_emb, freeze=freeze, padding_idx=self.padding_idx)


class SingleLayerRNN(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, num_cls:int, padding_idx:int) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.RNN = nn.RNN(embedding_dim, embedding_dim, num_layers=1, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(embedding_dim*2, num_cls)
    def forward(self, text_ids:torch.Tensor, length:List[int]) -> torch.Tensor:
        # TODO 4 实现输入词语id, 查表得到word embedding,
        #print(text_ids.shape)
        embeded = self.embedding(text_ids)
        sentenced_embedding = F.avg_pool2d(embeded,(embeded.shape[1],1))
        rnn_output,hidden =  self.RNN(sentenced_embedding)
        #output = torch.cat((rnn_output[-2,:,:], rnn_output[-1,:,:]), dim=1) # [batch size, hid dim * num directions]
        #print(rnn_output.shape)
        return self.fc(rnn_output.squeeze(1))
        # 将句内word embedding平均得到sentence embedding,
        # 最后将sentence embedding 过全连接层进行分类的过程
        # 注意batch内各个序列的有效长度不同，最后一个时间步对batch内不同序列是不同的
        # (当然你也可以使用pack_padded_sequence 函数, 绕过以上问题，详见
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html)
        pass

    def _init_embedding(self, pretrain_emb:torch.Tensor, freeze:bool=True) -> None:
        assert pretrain_emb.shape == self.embedding.weight.shape, 'invalid table shape'
        self.embedding = nn.Embedding.from_pretrained(pretrain_emb, freeze=freeze, padding_idx=self.padding_idx)

class SingleLayerLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        # TODO 5 可选, 实现LSTM版本的分类器
        pass

    def forward(self, text,length):
        embedded = self.dropout(self.embedding(text))  # [sent len, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded)
        output = self.fc(output)
        #print(output.shape)
        output = F.avg_pool2d(output,(output.shape[1],1))
        #print(output.shape)
        return output.squeeze(1)



if __name__ == '__main__':
    # TODO 6 可选, 验证实现的模型的输入输出的维度的正确性
    emb = torch.randn((10, 300))
    #print(emb.shape)
    model = WordAverage(emb.shape[0], emb.shape[1], num_cls=2, padding_idx=0)
    #model = SingleLayerRNN(emb.shape[0],emb.shape[1],num_cls=2,padding_idx=0)
    #model = SingleLayerLSTM(emb.shape[0],emb.shape[1],hidden_dim = emb.shape[1],output_dim = 2,n_layers=1,bidirectional=2,dropout = 0.8,pad_idx=0)
    sentence1 = torch.tensor([5, 6, 7])
    sentence2 = torch.tensor([1, 2, 3, 4, 3])
    

    dummy_input = [sentence1, sentence2]
    seq_len = [s.size(0) for s in dummy_input] # 获取数据真实的长度
    dummy_input = pad_sequence(dummy_input, batch_first=True)
    # dummy_input = pack_padded_sequence(dummy_input, seq_len, batch_first=True)
    print(dummy_input)
    output = model(dummy_input.view(dummy_input.shape[0],dummy_input.shape[1]), seq_len)
    print(model)
    print(f"input shape:{dummy_input.shape}, output shape:{output.shape}")