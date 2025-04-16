# emotion_classifier_8emo.py
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import os
import json

# 固定随机种子保证可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_data(data_path="emotion_data_manual_cleaned.csv"):
    """加载并预处理数据，强制使用12种情绪标签"""
    # 明确定义12类情绪及其顺序（关键！）
    TARGET_EMOTIONS = ["高兴", "厌恶", "害羞", "害怕", 
                      "生气", "认真", "紧张", "慌张", 
                      "疑惑", "兴奋", "无奈", "担心"]
    
    # 加载数据并筛选目标情绪
    data = pd.read_csv(data_path)
    data = data[data["label"].isin(TARGET_EMOTIONS)]
    data["text"] = data["text"].astype(str)
    
    # 数据统计
    print("\n=== 数据统计 ===")
    print("总样本数:", len(data))
    print("类别分布:\n", data["label"].value_counts())
    
    # 使用固定顺序的标签编码器
    label_encoder = LabelEncoder()
    label_encoder.fit(TARGET_EMOTIONS)  # 强制按定义顺序编码
    
    # 划分数据集（保证测试集包含所有类别）
    min_test_size = len(TARGET_EMOTIONS) / len(data)
    test_size = max(0.2, min(min_test_size, 0.3))
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data["text"].tolist(),
        data["label"].tolist(),
        test_size=test_size,
        stratify=data["label"],
        random_state=SEED
    )
    
    # 编码标签
    train_labels_encoded = label_encoder.transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    
    print(f"\n划分结果: 训练集={len(train_texts)}, 测试集={len(test_texts)}")
    print("测试集类别分布:", pd.Series(test_labels).value_counts())
    print("标签映射:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    
    return train_texts, test_texts, train_labels_encoded, test_labels_encoded, label_encoder

class EmotionDataset(torch.utils.data.Dataset):
    """自定义数据集类"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_and_evaluate():
    """训练和评估12类情绪分类模型"""
    # 1. 加载数据
    train_texts, test_texts, train_labels, test_labels, label_encoder = load_data()
    
    # 2. 初始化模型和分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", use_fast=False)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=12,  # 12分类任务
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )
    
    # 3. 数据编码
    print("\nTokenizing数据...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    # 4. 创建数据集
    train_dataset = EmotionDataset(train_encodings, train_labels)
    test_dataset = EmotionDataset(test_encodings, test_labels)
    
    # 5. 训练配置（优化后的超参数）
    training_args = TrainingArguments(
        output_dir="./results_8emo",        # 输出目录
        num_train_epochs=8,                # 12分类需要更多epoch
        per_device_train_batch_size=16,    # 适当增大batch size
        per_device_eval_batch_size=32,
        learning_rate=2e-5,               # 更小的学习率
        weight_decay=0.02,                 # 更强的正则化
        warmup_ratio=0.1,                  # 10%的warmup步数
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted", # 按加权F1选择最佳模型
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),    # 自动启用混合精度
        report_to="none"                   # 禁用wandb等记录
    )
    
    # 6. 自定义评估指标
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        report = classification_report(
            labels, preds, 
            target_names=label_encoder.classes_, 
            output_dict=True,
            zero_division=0
        )
        return {
            "f1_weighted": report["weighted avg"]["f1-score"],
            "accuracy": report["accuracy"]
        }
    
    # 7. 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("\n开始训练...")
    trainer.train()
    
    # 8. 最终评估
    print("\n=== 测试集性能 ===")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    print(classification_report(
        test_labels,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    # 9. 保存模型和配置
    output_dir = "./emotion_model_12emo"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 保存标签映射
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump({
            "id2label": {i: label for i, label in enumerate(label_encoder.classes_)},
            "label2id": {label: i for i, label in enumerate(label_encoder.classes_)}
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n模型和配置已保存到 {output_dir}")

if __name__ == "__main__":
    train_and_evaluate()