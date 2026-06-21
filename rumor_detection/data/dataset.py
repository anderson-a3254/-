"""
数据集加载模块
【来自MD】tokenize, build_vocab, encode, RumorDataset
"""
import re
import torch
from collections import Counter
from torch.utils.data import Dataset


# ============================================================
# 以下函数【来自MD】— 简单分词器 & 词表构建 & 编码
# ============================================================
def tokenize(text):
    """【来自MD】简单分词：小写 + 提取单词"""
    return re.findall(r'\w+', text.lower())


def build_vocab(texts, min_freq=2):
    """【来自MD】构建词表，保留词频 >= min_freq 的词"""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for w, c in counter.items():
        if c >= min_freq:
            vocab[w] = idx
            idx += 1
    return vocab


def encode(text, vocab, max_len=64):
    """【来自MD】将文本转为ID序列并padding/截断"""
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(ids) < max_len:
        ids += [vocab['<PAD>']] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


# ============================================================
# 【来自MD】RumorDataset — 用于 BiGRU 模型的数据集
# ============================================================
class RumorDataset(Dataset):
    """【来自MD】谣言检测数据集，返回文本ID和标签"""
    def __init__(self, df, vocab, max_len=64):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(encode(self.texts[idx], self.vocab, self.max_len), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y
