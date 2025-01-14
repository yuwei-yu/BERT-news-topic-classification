import torch

Config = {
    "model": "bert",
    "model_path": "bert",
    "pretrain_model_path": r"/Users/yuwei/Downloads/week6 语言模型和预训练/bert-base-chinese",
    "train_data_path": r"./data/train_tag_news.json",
    "valid_data_path": r"./data/valid_tag_news.json",
    "vocab_path": "./chars.txt",
    "learn_rate": 2e-5,
    "batch_size": 32,
    "epochs": 100,
    "max_len": 128,
    "hidden_size": 768,
    "optimizer": "adam",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "num_layer": 2,
    "max_length": 30,
    "pooling_type": "max",
    "class_num":18
}
