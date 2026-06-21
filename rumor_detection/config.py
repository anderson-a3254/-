"""
全局配置文件
【新增】统一管理所有路径、超参数和模型设置
"""
import os
import torch


# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集路径（rumer2026 文件夹在项目目录内）
DATA_DIR = os.path.join(BASE_DIR, 'rumer2026')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
VAL_PATH = os.path.join(DATA_DIR, 'val.csv')

# 模型保存目录
SAVE_DIR = os.path.join(BASE_DIR, 'saved')
os.makedirs(SAVE_DIR, exist_ok=True)

# 模型 & 词表文件路径
MODEL_PATH = os.path.join(SAVE_DIR, 'bigru.pt')
VOCAB_PATH = os.path.join(SAVE_DIR, 'vocab.pkl')


# ============ 设备配置 ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============ BiGRU 模型配置（来自MD文件）============
BATCH_SIZE = 32
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
EPOCHS = 10
MAX_LEN = 64


# ============ SJTU LLM API 配置 ============
# 依据 SJTU 平台文档：https://models.sjtu.edu.cn
# 模型仅限校内网络调用，数据不出校
SJTU_API_KEY = os.environ.get('SJTU_API_KEY', 'sk-1wsk1_inJiQK5jp3GmlEWQ')
SJTU_API_URL = 'https://models.sjtu.edu.cn/api/v1/chat/completions'
SJTU_MODEL_NAME = 'deepseek-chat'
LLM_MAX_TOKENS = 300
LLM_TEMPERATURE = 0.3
LLM_MAX_RETRIES = 5

