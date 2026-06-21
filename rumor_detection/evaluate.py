"""
验证集评估脚本
【新增】在 val.csv 上运行完整评估，输出准确率、混淆矩阵、分类报告

运行方式（在项目根目录执行）：
    python evaluate.py
"""
import sys
import os
import pickle

# 修复导入路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from config import *
from data.dataset import RumorDataset
from models.classifier import BiGRU


def evaluate_on_val(model_path=None, vocab_path=None):
    """在 val.csv 上完整评估"""
    print("=" * 60)
    print("  val.csv 验证集评估 (BiGRU)")
    print("=" * 60)

    # 加载数据
    val_df = pd.read_csv(VAL_PATH)
    print(f"\n验证集大小: {len(val_df)}")
    print(f"标签分布: 非谣言={sum(val_df['label']==0)}, 谣言={sum(val_df['label']==1)}")

    # 加载词表
    if vocab_path is None:
        vocab_path = VOCAB_PATH
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # 创建 DataLoader
    val_set = RumorDataset(val_df, vocab, max_len=MAX_LEN)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 加载模型
    model = BiGRU(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    if model_path is None:
        model_path = MODEL_PATH
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"模型加载: {model_path}\n")

    # 预测
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc='评估中'):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    all_preds = [int(p) for p in all_preds]
    all_labels = [int(l) for l in all_labels]

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*50}")
    print(f"  验证集准确率: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*50}")

    print(f"\n混淆矩阵:")
    print(f"              预测非谣言  预测谣言")
    print(f"  实际非谣言     {cm[0][0]:5d}      {cm[0][1]:5d}")
    print(f"  实际谣言       {cm[1][0]:5d}      {cm[1][1]:5d}")

    print(f"\n分类报告:")
    print(classification_report(
        all_labels, all_preds,
        target_names=['非谣言 (0)', '谣言 (1)'],
        digits=4
    ))

    # 每个 event 的准确率（泛化能力分析）
    val_df_copy = val_df.copy()
    val_df_copy['pred'] = all_preds
    print("\n各事件类别准确率（泛化能力评估）:")
    for event in sorted(val_df_copy['event'].unique()):
        sub = val_df_copy[val_df_copy['event'] == event]
        if len(sub) > 0:
            event_acc = (sub['label'] == sub['pred']).mean()
            print(f"  Event {event}: {event_acc:.4f} ({len(sub)}条)")

    return acc


if __name__ == '__main__':
    evaluate_on_val()
