import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from data_utils import WaiMaiTokenizer, WaiMaiMiniDataset, _waimai_collate_fn
from model import SingleLayerLSTM, SingleLayerRNN, WordAverage
from criterion import CrossEntropyWithLogit, Accuracy

import matplotlib.pyplot as plt
# TODO 8 在主脚本中调用之前已经写好的各个组件完成训练

def train(model:nn.Module,
          data_loader:DataLoader,
          optimizer:torch.optim.Optimizer,
          criterion:nn.Module):

    model.train()
    for batch_id, (text, length, label) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(text, length)
        #print(output.shape)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        print(f"epoch:{epoch} - batch:{batch_id}, loss:{loss.item():.4f}")

def valid_test(model:nn.Module,
               data_loader:DataLoader,
               metric:nn.Module):
    model.eval()
    with torch.no_grad():
        sum_metric_result = 0
        for batch_id, (text, length, label) in enumerate(data_loader):
            output = model(text, length)
            metric_result = metric(output, label)
            sum_metric_result += metric_result * output.shape[0]
        
        average_metric_result = sum_metric_result / (data_loader.batch_size * len(data_loader))
        print(f"Avg {metric}: {average_metric_result:.4f}")
        return average_metric_result
if __name__ == '__main__':
    vec_table, emb = WaiMaiTokenizer._build_from_word_vec_table('./data/waimai_mini/tiny_word_vec.json')
    tokenizer = WaiMaiTokenizer(vec_table, vec_table['stoi']['<UNK>'])
    train_set = WaiMaiMiniDataset(data_dir='./data/waimai_mini', train=True, tokenizer=tokenizer)

    train_set, valid_set = random_split(train_set, [int(0.8 * len(train_set)), len(train_set) - int(0.8 * len(train_set))])
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, collate_fn=_waimai_collate_fn)
    valid_dataloader = DataLoader(valid_set, batch_size=128, shuffle=True, num_workers=4, collate_fn=_waimai_collate_fn)

    test_set = WaiMaiMiniDataset(data_dir='./data/waimai_mini', train=False, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4, collate_fn=_waimai_collate_fn)
    #model = WordAverage(emb.shape[0], emb.shape[1], num_cls=2, padding_idx=vec_table['stoi']['<PAD>'])
    model = SingleLayerRNN(emb.shape[0], emb.shape[1], num_cls=2, padding_idx=vec_table['stoi']['<PAD>'])
    #model = SingleLayerLSTM(vocab_size = emb.shape[0], embedding_dim = emb.shape[1],hidden_dim=emb.shape[1], output_dim=2, n_layers = 4,bidirectional=False,dropout=0.5,pad_idx=vec_table['stoi']['<PAD>'])

    # model._init_embedding(emb, freeze=False)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)

    criterion = CrossEntropyWithLogit()
    acc_metric = Accuracy()
    best_acc = 0
    dev_acc = []
    epochs = []
    for epoch in range(100):
        train(model, train_dataloader, optimizer, criterion)

        print(f"epoch:{epoch}, on validation set:")
        acc = valid_test(model, valid_dataloader, acc_metric)
        # TODO 9 实现你的停止训练的策略
        dev_acc.append(acc)
        epochs.append(epoch)
        if acc > best_acc:
            best_acc = acc
            continue
        else:
            if epoch >= 50:
                break
    print("on testing set:")
    valid_test(model, test_dataloader, acc_metric)
    plt.figure('dev上的准确率变化')
    plt.plot(epochs,dev_acc)
    plt.legend()
    plt.grid()
    plt.show()