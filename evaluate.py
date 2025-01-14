import torch
from loader import load_data


class Evaluator:
    """
    用于评估模型性能的类。

    Attributes:
        config (dict): 配置信息。
        model: 模型实例。
        logger: 日志记录器。
        device: 设备信息，用于模型和数据的加载。
        valid_data: 验证数据集。
        stats_dict (dict): 用于统计验证结果，包含"correct"和"wrong"两个键。
    """

    def __init__(self, config, model, logger):
        """
        初始化Evaluator类。

        Args:
            config (dict): 配置信息。
            model: 模型实例。
            logger: 日志记录器。
        """
        self.config = config
        self.model = model
        self.logger = logger
        self.device = torch.device(config["device"])
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    def evaluate(self, epoch):
        """
        评估模型在特定轮次的性能。

        Args:
            epoch (int): 当前轮次。

        Returns:
            float: 模型的准确率。
        """
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

        for index, data in enumerate(self.valid_data):
            data = [d.to(self.device) for d in data]
            input_ids, label = data
            with torch.no_grad():
                predict = self.model(input_ids)
                self.write_state(label, predict)
        return self.print_state()

    def write_state(self, label, predict_result):
        """
        更新验证结果状态。

        Args:
            label (Tensor): 真实标签。
            predict_result (Tensor): 预测结果。
        """
        assert len(label) == len(predict_result)
        for true_label, predict_label in zip(label, predict_result):
            predict_label = predict_label.argmax(dim=-1)
            if int(predict_label) == int(true_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1

    def print_state(self):
        """
        打印验证结果。

        Returns:
            float: 模型的准确率。
        """
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
