import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.optim import Adam,SGD
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
from config import Config
import time
class Model(nn.Module):
    """
    用于文本分类的BERT模型。

    该模型首先通过预训练的BERT模型对输入的文本进行编码，然后根据配置的池化类型
    对编码结果进行池化操作，最后通过全连接层将池化后的结果映射到类别空间。

    参数:
    - config (dict): 包含模型配置的字典，必须包含预训练模型路径、类别数量和池化类型。
    """

    def __init__(self, config):
        super(Model, self).__init__()
        # 确保配置项存在且类型正确
        required_keys = [ "class_num", "pooling_type"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        # 修改bert的layer_num

        print(self.bert.config)
        self.hidden_size = self.bert.config.hidden_size
        self.num_class = config["class_num"]
        self.pooling_type = config["pooling_type"]


        # 初始化分类层
        self.classify = nn.Linear(self.hidden_size, self.num_class)
        # 初始化损失函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, target=None):
        """
        模型的前向传播函数。

        参数:
        - input_ids (torch.Tensor): 输入的文本ID序列，形状为(batch_size, sequence_length)。
        - target (torch.Tensor, 可选): 目标类别标签，形状为(batch_size)。

        返回:
        - torch.Tensor: 如果target未提供，则返回模型的预测结果；如果提供了target，则返回损失值。
        """
        # 验证 input_ids 的形状
        if input_ids.dim() != 2:
            raise ValueError("input_ids should be a 2D tensor with shape (batch_size, sequence_length)")

        # 通过BERT模型获取序列的编码输出
        seq_output, _ = self.bert(input_ids)
        print(seq_output.shape)
        # 初始化池化层
        if self.pooling_type == "max":
            self.pooling_layer = nn.MaxPool1d(seq_output.shape[1])
        elif self.pooling_type == "avg":
            self.pooling_layer = nn.AvgPool1d(seq_output.shape[1])
        else:
            raise ValueError("Unsupported pooling type. Choose 'max' or 'avg'.")

        # 执行池化操作
        x = self.pooling_layer(seq_output.permute(0, 2, 1)).squeeze(-1)
        # 通过全连接层进行分类
        predict = self.classify(x)

        if target is None:
            return predict
        print(predict.shape,target.shape)
        # 计算损失
        return self.loss(predict, target.squeeze())

#优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
