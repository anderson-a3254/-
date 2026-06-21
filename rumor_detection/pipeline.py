"""
主预测管线
【来自MD参考】类结构参考 MD 文件的 RumourDetectClass
【新增】BiGRU 分类器 + SJTU LLM 解释器 = 完整的「检测+解释」输出

运行方式（在项目根目录执行）：
    python pipeline.py
"""
import sys
import os

# 修复导入路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import pickle
import re

from config import *
from data.dataset import encode
from models.classifier import BiGRU
from explain.llm_explainer import LLMExplainer


class RumorDetectionPipeline:
    """
    【核心类】可解释谣言检测管线

    输入：一段推文文本
    输出1：二分类结果（0=非谣言, 1=谣言）+ 置信度
    输出2：判断依据（由 LLM 生成）

    使用方式：
        pipeline = RumorDetectionPipeline()
        result = pipeline.predict("BREAKING: ...")
        # result = {
        #     'prediction': 1,
        #     'label': '谣言',
        #     'confidence': 0.9234,
        #     'explanation': '该推文使用了BREAKING等紧急词汇...'
        # }
    """

    def __init__(self, model_path=None, vocab_path=None, use_llm=False):
        """
        初始化管线
        Args:
            model_path: 模型权重路径
            vocab_path: 词表路径
            use_llm: 是否启用 LLM（需要 API key + 校内网络）
        """
        self.device = DEVICE
        self.use_llm = use_llm

        if model_path is None:
            model_path = MODEL_PATH
        if vocab_path is None:
            vocab_path = VOCAB_PATH

        # ---- 加载词表 ----
        print('[Pipeline] 加载词表 ...')
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        print(f'[Pipeline] 词表大小: {len(self.vocab)}')

        # ---- 加载 BiGRU 模型 ----
        print('[Pipeline] 加载 BiGRU 模型 ...')
        self.model = BiGRU(len(self.vocab), EMBEDDING_DIM, HIDDEN_DIM)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f'[Pipeline] 模型加载成功: {model_path}')

        # ---- 加载 LLM 解释器 ----
        if self.use_llm:
            print('[Pipeline] 加载 LLM 解释器 (deepseek-chat) ...')
            self.explainer = LLMExplainer()
        else:
            self.explainer = None
            print('[Pipeline] 使用规则降级方案（无需 API）')

        print('[Pipeline] 初始化完成！\n')

    def preprocess(self, text):
        """【来自MD参考】文本基本清理"""
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def classify(self, text):
        """
        分类预测
        Args:
            text: 输入文本
        Returns:
            dict: prediction, label, confidence
        """
        text = self.preprocess(text)
        x = torch.tensor(
            encode(text, self.vocab, MAX_LEN),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(x)          # shape: (1,)
            prob = torch.sigmoid(logit).item()  # 谣言概率
            pred = 1 if prob > 0.5 else 0
            confidence = prob if pred == 1 else (1 - prob)

        return {
            'prediction': pred,
            'label': '谣言' if pred == 1 else '非谣言',
            'confidence': round(confidence, 4),
            'prob_rumor': round(prob, 4),
        }

    def predict(self, text):
        """
        完整预测：分类 + 判断依据

        Args:
            text: 输入推文文本
        Returns:
            dict: prediction, label, confidence, explanation
        """
        result = self.classify(text)

        # 生成判断依据
        if self.use_llm and self.explainer is not None:
            explanation = self.explainer.generate(text, result['prediction'])
        else:
            explanation = LLMExplainer()._fallback_explanation(text, result['prediction'])

        result['explanation'] = explanation
        return result


# ============================================================
# 交互式测试
# ============================================================
def interactive_test():
    print("=" * 60)
    print("  可解释谣言检测系统 (BiGRU + deepseek-chat)")
    print("=" * 60)
    print(f"  设备: {DEVICE}")
    print()
    print("  使用说明:")
    print("  - 输入推文文本，回车查看检测结果")
    print("  - 输入 'quit' 退出")
    print("  - 如需 LLM 解释，先设置 API key 并改 use_llm=True")
    print()

    pipeline = RumorDetectionPipeline(use_llm=False)

    print("输入推文进行检测（输入 'quit' 退出）：\n")

    while True:
        text = input("推文 > ").strip()
        if text.lower() == 'quit':
            print("再见！")
            break
        if not text:
            continue

        result = pipeline.predict(text)

        print(f"\n{'='*55}")
        print(f"  检测结果: {result['label']}")
        print(f"  置信度:   {result['confidence']}")
        print(f"  谣言概率: {result['prob_rumor']}")
        print(f"{'─'*55}")
        print(f"  判断依据: {result['explanation']}")
        print(f"{'='*55}\n")


# ============================================================
# 批量测试 — 在 val.csv 上跑几条样例
# ============================================================
def batch_test(n=5):
    """在 val.csv 上测试前 n 条样本"""
    import pandas as pd

    print(f"从 val.csv 加载前 {n} 条样本进行测试...\n")

    pipeline = RumorDetectionPipeline(use_llm=False)
    val_df = pd.read_csv(VAL_PATH)

    for i in range(min(n, len(val_df))):
        text = val_df.iloc[i]['text']
        true_label = val_df.iloc[i]['label']

        result = pipeline.predict(text)

        correct = '✓' if result['prediction'] == true_label else '✗'
        print(f"[{i+1}] {correct} 真实: {'谣言' if true_label else '非谣言'} | "
              f"预测: {result['label']} (置信度: {result['confidence']})")
        print(f"    推文: {text[:80]}...")
        print(f"    依据: {result['explanation'][:80]}...")
        print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='可解释谣言检测管线')
    parser.add_argument('--batch', type=int, default=0,
                        help='批量测试 n 条样本（默认0=交互模式）')
    args = parser.parse_args()

    if args.batch > 0:
        batch_test(args.batch)
    else:
        interactive_test()
