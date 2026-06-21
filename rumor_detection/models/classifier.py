"""
分类模型定义
【来自MD】BiGRU — MD 文件中的双向GRU模型
"""
import torch
import torch.nn as nn


class BiGRU(nn.Module):
    """【来自MD】双向GRU谣言检测模型"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bigru = nn.GRU(
            embedding_dim, hidden_dim,
            batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.bigru(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        out = self.fc(h)
        return out.squeeze(1)
