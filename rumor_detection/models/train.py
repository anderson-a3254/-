"""
模型训练脚本
【来自MD】训练 BiGRU 模型，保存模型权重 + 词表

运行方式（在项目根目录执行）：
    python models/train.py
"""
import sys
import os
import pickle

# ============================================================
# 【新增】修复导入路径
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from config import *
from data.dataset import RumorDataset, build_vocab
from models.classifier import BiGRU


# ============================================================
# 【来自MD】评估函数
# ============================================================
def evaluate(model, loader):
    """【来自MD】评估函数，返回准确率"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


# ============================================================
# 【基于MD的main()扩展】训练函数
# ============================================================
def train():
    """
    【基于MD的main()函数扩展】
    训练 BiGRU 模型，保存模型权重 + 词表（供 pipeline.py 推理使用）
    """
    print("=" * 60)
    print("  训练 BiGRU 谣言检测模型")
    print("=" * 60)

    # 读取数据
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    print(f"训练集: {len(train_df)} 条, 验证集: {len(val_df)} 条")
    print(f"标签分布 — 训练集: 谣言={sum(train_df['label']==1)}, 非谣言={sum(train_df['label']==0)}")
    print(f"标签分布 — 验证集: 谣言={sum(val_df['label']==1)}, 非谣言={sum(val_df['label']==0)}")

    # 构建词表
    vocab = build_vocab(train_df['text'])
    print(f"词表大小: {len(vocab)}")

    # 构建数据集
    train_set = RumorDataset(train_df, vocab, max_len=MAX_LEN)
    val_set = RumorDataset(val_df, vocab, max_len=MAX_LEN)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # 初始化模型
    model = BiGRU(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    print(f"设备: {DEVICE}, 超参数: embed={EMBEDDING_DIM}, hidden={HIDDEN_DIM}, epochs={EPOCHS}")
    print()

    # 训练循环
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader)
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1:2d}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}',
              '(*)' if val_acc > best_acc else '')

        if val_acc > best_acc:
            best_acc = val_acc
            # 保存模型权重
            torch.save(model.state_dict(), MODEL_PATH)
            # 【新增】保存词表，供推理时使用
            with open(VOCAB_PATH, 'wb') as f:
                pickle.dump(vocab, f)

    print(f'\n最佳验证准确率: {best_acc:.4f}')
    print(f'模型已保存: {MODEL_PATH}')
    print(f'词表已保存: {VOCAB_PATH}')

    # 用最佳模型做最终评估
    model.load_state_dict(torch.load(MODEL_PATH))
    final_acc = evaluate(model, val_loader)
    print(f'最终验证准确率: {final_acc:.4f} ({final_acc*100:.2f}%)')

    return model


if __name__ == '__main__':
    train()
