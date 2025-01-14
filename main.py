import logging
import numpy as np
import torch
from config import Config
from loader import load_data
from model import Model
from model import choose_optimizer
import os
from evaluate import Evaluator
import random
from utils import *
from time import time

# 初始化日志配置，设置日志级别和格式
# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

# 设置随机种子以确保结果的可复现性
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    """
    主函数，负责模型的训练和保存
    参数:
        config (dict): 包含配置信息的字典
    返回:
        acc (float): 最后一个epoch的准确率
    """
    # 确保模型保存路径存在
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])

    # 选择设备，优先使用GPU
    device = torch.device(Config["device"])

    # 初始化模型并将其移动到指定设备
    model = Model(Config).to(device)
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 根据配置选择优化器
    optimizer = choose_optimizer(Config, model)

    # 初始化评估器
    evaluator = Evaluator(Config, model, logging)
    model_train_logger = {}
    model_train_epoch_logger = []
    final_acc = 0
    # 开始训练
    for epoch in range(config['epochs']):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        # 迭代训练数据
        for index, batch_data in enumerate(train_data):
            batch_data = [data.to(device) for data in batch_data]
            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            # 定期输出loss信息
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        # 每个epoch结束时进行评估
        acc = evaluator.evaluate(epoch)
        model_train_epoch_logger.append({
            "epoch": epoch,
            "loss": np.mean(train_loss),
            "accuracy": acc,
        })
        final_acc = acc

    # 保存模型权重
    model_path = os.path.join(config["model_path"], "model.pth")
    torch.save(model.state_dict(), model_path)
    model_train_logger['model'] = config["model_type"]
    model_train_logger['learn_rate'] = config["learning_rate"]
    model_train_logger['epochs'] = config["epochs"]
    model_train_logger["layers"] = config["num_layer"]
    model_train_logger['accuracy'] = final_acc
    return model_train_logger, model_train_epoch_logger

if __name__ == "__main__":
    # # 调用主函数开始训练
    # main(Config)

    # 以下代码为不同模型和超参数的网格搜索，已注释
    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    model_train_loggers = []
    best_acc = 0
    best_train_logger = []
    for lr in [1e-4, 1e-5]:
        Config["learning_rate"] = lr
        for pooling_style in ["avg", 'max']:
            Config["pooling_style"] = pooling_style
            for batch_size in [64, 128]:
                Config["batch_size"] = batch_size
                model_train_logger, epoch_train_data = main(Config)
                model_train_loggers.append(model_train_logger)
                if model_train_logger['accuracy'] > best_acc:
                    best_acc = model_train_logger['accuracy']
                    best_train_logger = model_train_logger
                print("当前配置：", Config)

    generate_performance_table(model_train_loggers)
    plot_training_progress({
        "best-result": best_train_logger
    })
