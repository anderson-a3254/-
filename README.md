## 项目结构

rumor_detection/
├── config.py # 全局配置（路径、超参数、API密钥）
├── requirements.txt # 依赖列表
├── README.md # 本文件
├── data/
│ └── dataset.py # 数据集加载（BiGRU + BERT）
├── models/
│ ├── classifier.py # 分类模型（BiGRU + BERT）
│ └── train.py # 训练脚本
├── explain/
│ └── llm_explainer.py # LLM解释生成
├── pipeline.py # 主预测管线
├── evaluate.py # 验证集评估
└── saved/ # 保存的模型权重

## 环境配置

```bash
# 1. 安装依赖
cd ..\rumor_detection

pip install -r requirements.txt

# 2. 确保路径在正确位置
# 默认路径：../rumer2026/train.csv 和 ../rumer2026/val.csv
# 如需修改，请编辑 config.py 中的 DATA_DIR

# 3. 开始使用
#  训练 BiGRU（会保存模型+词表到 saved/）
python models/train.py
#  评估
python evaluate.py
#  交互测试（先用 use_llm=False）
python pipeline.py
#  批量测试 val.csv 前10条
python pipeline.py --batch 10
#  如需 LLM 解释：设置 API key，改 pipeline.py 中 use_llm=True

```
