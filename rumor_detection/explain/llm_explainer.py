"""
LLM 解释生成模块
【新增】使用 SJTU 大模型 API（deepseek-chat）生成谣言检测的判断依据
API格式参考：https://models.sjtu.edu.cn
"""
import time
import requests
from config import (
    SJTU_API_KEY, SJTU_API_URL, SJTU_MODEL_NAME,
    LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_MAX_RETRIES
)


class LLMExplainer:
    """
    【新增】基于SJTU deepseek-chat API的谣言判断依据生成器

    API 格式说明（来自 SJTU 平台文档）：
    - URL: https://models.sjtu.edu.cn/api/v1/chat/completions
    - Authorization: Bearer your-api-key
    - model: deepseek-chat（或 deepseek-reasoner）
    - 仅限校内网络调用
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or SJTU_API_KEY
        self.api_url = SJTU_API_URL
        self.model_name = SJTU_MODEL_NAME
        self.max_tokens = LLM_MAX_TOKENS
        self.temperature = LLM_TEMPERATURE
        self.max_retries = LLM_MAX_RETRIES

    def _build_prompt(self, text, predicted_label):
        """
        【新增】构建发送给 LLM 的 prompt

        设计思路：
        - system 角色设定为「谣言检测分析专家」
        - user 消息包含推文内容和模型判定结果
        - 引导 LLM 从语言特征、事实依据、情绪化程度等角度分析
        """
        label_str = '谣言（不可信信息）' if predicted_label == 1 else '非谣言（可信信息）'

        return {
            'system': (
                '你是一个专业的社交媒体谣言检测分析专家。'
                '你的任务是根据推文的语言特征，解释为什么一条推文被判定为谣言或非谣言。'
                '请用简洁的2-3句话给出判断依据，不需要重复推文内容。'
            ),
            'user': (
                f'请分析以下推文，并解释为什么它应该被判定为"{label_str}"。\n\n'
                f'【推文内容】：{text}\n\n'
                f'【模型检测结果】：{label_str}\n\n'
                f'请从以下角度简要分析：\n'
                f'1. 推文是否包含情绪化、煽动性语言或夸张表达？\n'
                f'2. 推文是否有可验证的事实依据或清晰的信息来源？\n'
                f'3. 推文的措辞特点（如使用BREAKING、大量感叹号、极端化表述等）\n\n'
                f'请直接给出判断依据。'
            )
        }

    def _call_api(self, messages):
        """
        【新增】调用 SJTU LLM API，带重试逻辑

        参考了 SJTU 平台提供的 Python 调用示例中的重试机制：
        - 最多重试 5 次
        - 4xx 错误（非429）不重试
        - 超时设置 (30, 600)
        """
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        data = {
            'model': self.model_name,
            'messages': messages,
            'stream': False,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.api_url, headers=headers, json=data,
                    timeout=(30, 600)
                )
            except requests.RequestException as e:
                print(f'[LLM] 请求异常(第{attempt}次): {e}')
                time.sleep(2)
                continue

            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content'].strip()

            print(f'[LLM] HTTP {resp.status_code} (第{attempt}次): {resp.text[:300]}')
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                break
            time.sleep(2)

        raise RuntimeError(f'LLM API 请求失败，已重试{self.max_retries}次')

    def generate(self, text, predicted_label):
        """
        【新增】调用 LLM 生成判断依据

        Args:
            text: 推文原文
            predicted_label: 模型预测的标签 (0=非谣言, 1=谣言)

        Returns:
            str: 判断依据文本
        """
        prompt = self._build_prompt(text, predicted_label)
        messages = [
            {'role': 'system', 'content': prompt['system']},
            {'role': 'user', 'content': prompt['user']},
        ]

        try:
            explanation = self._call_api(messages)
            return explanation
        except Exception as e:
            print(f'[LLM] 调用失败，使用规则降级方案: {e}')
            return self._fallback_explanation(text, predicted_label)

    def _fallback_explanation(self, text, predicted_label):
        """
        【新增】降级方案 — 当 LLM 不可用时，基于规则生成简单解释
        确保即使 API 不可用，系统仍能输出判断依据
        """
        text_lower = text.lower()

        if predicted_label == 1:
            reasons = []
            if any(w in text_lower for w in ['breaking', 'urgent', 'alert']):
                reasons.append('使用紧急/突发性词汇(BREAKING/URGENT)制造紧迫感')
            if text.count('!') >= 2:
                reasons.append('大量使用感叹号，情绪化表达明显')
            if any(w in text_lower for w in ['shocking', 'unbelievable', 'disgusting', 'worst', 'never']):
                reasons.append('包含极端化/煽动性词汇')
            if not reasons:
                reasons.append('缺乏可验证的信息来源，措辞具有煽动性')
            return '；'.join(reasons) + '。综合判断：该推文具有谣言特征。'
        else:
            return '该推文语言相对客观，未发现明显的谣言特征标记（如极端化措辞、情绪煽动等），内容具有一定的可验证性。'


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    explainer = LLMExplainer()

    print('测试1 - 谣言推文:')
    test_rumor = "BREAKING: Shocking news! The government is hiding the truth!!! #WakeUp"
    print(f'  推文: {test_rumor}')
    print(f'  解释: {explainer.generate(test_rumor, 1)}')
    print()

    print('测试2 - 非谣言推文:')
    test_non_rumor = "The city council meeting will be held at 7pm tomorrow at City Hall."
    print(f'  推文: {test_non_rumor}')
    print(f'  解释: {explainer.generate(test_non_rumor, 0)}')
