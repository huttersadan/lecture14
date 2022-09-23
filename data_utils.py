import torch
import json
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Callable, Tuple, Optional, Dict, List

class WaiMaiTokenizer(object):
    # TODO 1 可选, 阅读或实现tokenizer, 其中应当包含 分词、停用词过滤、词语转id等内容
    def __init__(self, vocab:Dict[str, dict], UNK_IDX:int) -> None:
        self.vocab = vocab
        self.UNK_IDX = UNK_IDX
        
    def __call__(self, text:str) -> List[int]:
        text = text.split(' ')
        ids = [self.vocab['stoi'].get(token, self.UNK_IDX) for token in text]
        return ids
    
    def _decode(self, ids:List[int]) -> str:
        tokens = [self.vocab['itos'].get(idx, '<UNK>') for idx in ids]
        return ''.join(tokens)

    @classmethod
    def _build_from_word_vec_table(cls, json_path:str,
                                   add_unk:bool=True,
                                   add_pad:bool=True,
                                   add_bos:bool=True,
                                   add_eos:bool=True
                         ) -> Tuple[Dict[str, dict], torch.Tensor]:
        with open(json_path, 'rt',encoding='utf-8') as f:
            word_vec_dict = json.load(f)
        word_list = []
        vec_list = []
        for word, vec in word_vec_dict.items():
            word_list.append(word)
            vec_list.append(vec)
            D = len(vec)
        if add_eos:
            word_list.insert(0, '<EOS>')
            vec_list.insert(0, torch.randn((D,)).numpy().tolist())
        if add_bos:
            word_list.insert(0, '<BOS>')
            vec_list.insert(0, torch.randn((D,)).numpy().tolist())
        if add_unk:
            word_list.insert(0, '<UNK>')
            vec_list.insert(0, torch.randn((D,)).numpy().tolist())
        if add_pad:
            word_list.insert(0, '<PAD>')
            vec_list.insert(0, torch.zeros((D,)).numpy().tolist())
        
        stoi = {word:idx for idx, word in enumerate(word_list)}
        itos = {idx:word for idx, word in enumerate(word_list)}
        emb_table = torch.stack([torch.tensor(vec) for vec in vec_list])
        return {'stoi': stoi, 'itos': itos}, emb_table
        
class WaiMaiMiniDataset(Dataset):
    def __init__(self, data_dir:str, train:bool=True, tokenizer:Optional[Callable[[str], List[int]]]=None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        data_path = Path(data_dir) / 'train.json' if train else Path(data_dir) / 'test.json'
        with open(data_path,encoding='utf-8') as f:
            self._meta_data = json.load(f)
  
    def __len__(self) -> int:
        return len(self._meta_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        meta_data = self._meta_data[index]

        text = meta_data['segment']
        if self.tokenizer is not None:
            text = self.tokenizer(text)
        label = meta_data['label']
        return text, label


def _waimai_collate_fn(data:Tuple[List[int], int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO 2 可选, 实现变长数据加载时,将不同序列补0补到相同长度并打包返回
    '''
    将变长的序列打包,短序列补0,并配置好长度
    '''
    #print(data)
    data.sort(key=lambda x:len(x[0]))
    text_tensor, label_tensor = list(zip(*data))
    max_length = max(len(text_tensor[i]) for i in range(len(text_tensor)))
    for inst in text_tensor:
        zero_list = [0]*(max_length-len(inst))
        inst.extend(zero_list)
    text_tensor = [torch.tensor(text) for text in text_tensor]
    seq_len = torch.tensor([s.shape[0] for s in text_tensor])
    text_tensor = pad_sequence(text_tensor, batch_first=True) 
    label_tensor = torch.tensor(label_tensor)
    return text_tensor, seq_len, label_tensor

if __name__ == '__main__':
    # TODO 3 可选, 验证你的Tokenizer, Dataset, DataLoader所需的collate_fn的正确性
    vec_table, emb = WaiMaiTokenizer._build_from_word_vec_table('data/waimai_mini/tiny_word_vec.json')
    tokenizer = WaiMaiTokenizer(vec_table, vec_table['stoi']['<UNK>'])
    dataset = WaiMaiMiniDataset(data_dir='data/waimai_mini', train=True, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=_waimai_collate_fn)
    for text_ids, label in dataset:
        print(text_ids)
        print(tokenizer._decode(text_ids))
        break
    
    # for text, label in enumerate(dataloader):
    #     print(text)
        
