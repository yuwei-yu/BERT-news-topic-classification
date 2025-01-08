import torch
from loader import load_data

class Evaluator:
    def __init__(self, config,model,logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.device = torch.device(config["device"])
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果
    def evaluate(self,epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

        for index, data in enumerate(self.valid_data):
            input_ids, label = data.to(self.device)
            with torch.no_grad():
                predict = self.model(input_ids)
                self.write_state(label,predict)
        return self.print_state()
    def write_state(self,label,predict_result):
        assert len(label)==len(predict_result)
        for true_label, predict_label in zip(label, predict_result):
            predict_label = predict_label.argmax(dim=-1)
            if int(predict_label) == int(true_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return
    def print_state(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)



