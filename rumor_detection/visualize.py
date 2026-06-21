"""
可视化分析脚本
【新增】生成训练曲线、混淆矩阵、事件准确率三张图
运行方式：python visualize.py
"""
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import pickle
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm

from config import *
from data.dataset import RumorDataset
from models.classifier import BiGRU


# ============================================================
# 图1: 训练曲线
# ============================================================
def plot_training_curve():
    """绘制 Loss 和 Accuracy 随 Epoch 变化的双轴曲线"""
    epochs = list(range(1, 11))
    loss = [0.5785, 0.4018, 0.3028, 0.2199, 0.1561,
            0.1043, 0.0622, 0.0434, 0.0347, 0.0254]
    val_acc = [0.7880, 0.8030, 0.8130, 0.8229, 0.8304,
               0.8229, 0.8005, 0.8005, 0.8130, 0.8204]

    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    # Loss 轴（左）
    color1 = '#E74C3C'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', color=color1, fontsize=12)
    line1, = ax1.plot(epochs, loss, 'o-', color=color1, linewidth=2.5,
                      markersize=7, markerfacecolor='white',
                      markeredgewidth=2, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 0.7)

    # Accuracy 轴（右）
    ax2 = ax1.twinx()
    color2 = '#2980B9'
    ax2.set_ylabel('Validation Accuracy', color=color2, fontsize=12)
    line2, = ax2.plot(epochs, val_acc, 's--', color=color2, linewidth=2.5,
                      markersize=7, markerfacecolor='white',
                      markeredgewidth=2, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.74, 0.86)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    # 标注最佳点
    best_epoch, best_acc = 5, 0.8304
    ax2.annotate(f'Best: Epoch {best_epoch}\nAcc = {best_acc:.2%}',
                 xy=(best_epoch, best_acc),
                 xytext=(best_epoch + 1.8, best_acc - 0.025),
                 arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2),
                 fontsize=11, color='#1E8449', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1',
                           edgecolor='#27AE60', alpha=0.8))

    # 图例
    lines = [line1, line2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right',
               fontsize=10)

    plt.title('BiGRU Training: Loss Curve & Validation Accuracy',
              fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'figure1_training_curve.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print('[√] 图1 已保存: figure1_training_curve.png')


# ============================================================
# 图2: 混淆矩阵
# ============================================================
def plot_confusion_matrix():
    """在 val.csv 上预测并绘制混淆矩阵"""
    # 加载数据
    val_df = pd.read_csv(VAL_PATH)
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    val_set = RumorDataset(val_df, vocab, max_len=MAX_LEN)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 加载模型
    model = BiGRU(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 预测
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    all_preds = [int(p) for p in all_preds]
    all_labels = [int(l) for l in all_labels]

    # 计算指标
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n{'='*50}")
    print(f"  验证集评估结果")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n{classification_report(all_labels, all_preds,
           target_names=['非谣言', '谣言'], digits=4)}")

    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

    # 标注数值
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i][j] > cm.max()/2 else '#2C3E50'
            ax.text(j, i, f'{cm[i][j]}', ha='center', va='center',
                    fontsize=28, fontweight='bold', color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['预测: 非谣言', '预测: 谣言'], fontsize=12)
    ax.set_yticklabels(['实际: 非谣言', '实际: 谣言'], fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')

    plt.title(f'Confusion Matrix (Acc={acc:.2%})',
              fontsize=14, fontweight='bold', pad=15)

    # 添加色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Count', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'figure2_confusion_matrix.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print('[√] 图2 已保存: figure2_confusion_matrix.png')

    return all_preds, all_labels, val_df


# ============================================================
# 图3: 各事件准确率
# ============================================================
def plot_event_accuracy(all_preds, all_labels, val_df):
    """按 event 分组绘制准确率柱状图"""
    val_df = val_df.copy()
    val_df['pred'] = all_preds

    event_names = {
        0: 'Gurlitt\n艺术纠纷',
        1: 'Ferguson\n社会事件',
        2: 'Ebola/Essien\n健康谣言',
        3: 'Prince\n娱乐演出',
        4: 'Germanwings\n空难',
        5: 'Sydney Siege\n人质事件',
        6: 'Ottawa\n枪击事件'
    }

    events_sorted = sorted(val_df['event'].unique())
    event_accs = []
    event_sizes = []
    for e in events_sorted:
        sub = val_df[val_df['event'] == e]
        acc = (sub['label'] == sub['pred']).mean()
        event_accs.append(acc)
        event_sizes.append(len(sub))
        print(f"  Event {e} ({event_names.get(e, '?')}): "
              f"Acc={acc:.4f}, n={len(sub)}")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    names = [event_names.get(e, f'Event {e}') for e in events_sorted]
    x = np.arange(len(names))
    overall_acc = sum(a*s for a, s in zip(event_accs, event_sizes)) / sum(event_sizes)

    # 颜色：高于平均绿色，低于平均橙色
    colors = ['#27AE60' if a >= overall_acc else '#E67E22' for a in event_accs]
    bars = ax.bar(x, event_accs, color=colors, edgecolor='white',
                  linewidth=1.5, width=0.6)

    # 整体准确率参考线
    ax.axhline(y=overall_acc, color='#E74C3C', linestyle='--',
               linewidth=2, label=f'Overall Acc = {overall_acc:.2%}')

    # 标注数值
    for bar, acc, size in zip(bars, event_accs, event_sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f'{acc:.1%}\n(n={size})', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0.6, 1.08)
    ax.legend(fontsize=11, loc='lower right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    plt.title('Accuracy by Event Category (Cross-Event Generalization)',
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'figure3_event_accuracy.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print('[√] 图3 已保存: figure3_event_accuracy.png')


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  生成报告可视化图表")
    print("=" * 60)

    # 图1: 训练曲线（使用硬编码的训练日志数据）
    plot_training_curve()

    # 图2 + 图3: 需要在 val.csv 上预测
    preds, labels, val_df = plot_confusion_matrix()

    # 图3: 各事件准确率
    plot_event_accuracy(preds, labels, val_df)

    print(f"\n所有图表已保存到: {PROJECT_ROOT}/")
    print("  - figure1_training_curve.png")
    print("  - figure2_confusion_matrix.png")
    print("  - figure3_event_accuracy.png")
