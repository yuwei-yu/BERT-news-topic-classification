
import pandas as pd
import matplotlib.pyplot as plt
import json

#
# # 假设这是模型训练参数和每个epoch的损失和准确率
# models_info = [
#     {
#         "model_name": "Model_A",
#         "epochs": 10,
#         "learning_rate": 0.001,
#         "final_loss": 0.25,
#         "final_accuracy": 0.92
#     },
#     {
#         "model_name": "Model_B",
#         "epochs": 15,
#         "learning_rate": 0.0005,
#         "final_loss": 0.35,
#         "final_accuracy": 0.88
#     }
# ]

# 将训练参数保存到文件
def save_json_file(models_info, filename="model_params.json"):
    with open(filename, 'w') as f:
        json.dump(models_info, f, indent=4)

# # 保存训练过程中的每个epoch的loss和accuracy
# epoch_data = {
#     "Model_A": [
#         {"epoch": 1, "loss": 0.6, "accuracy": 0.75},
#         {"epoch": 2, "loss": 0.5, "accuracy": 0.80},
#         # 继续存储每个epoch的数据
#     ],
#     "Model_B": [
#         {"epoch": 1, "loss": 0.7, "accuracy": 0.70},
#         {"epoch": 2, "loss": 0.6, "accuracy": 0.75},
#         # 继续存储每个epoch的数据
#     ]
# }


# 读取模型训练参数
def load_json_file(filename="model_params.json"):
    with open(filename, 'r') as f:
        return json.load(f)

# 生成模型性能对比表格
def generate_performance_table(models_info):
    df = pd.DataFrame(models_info)
    df.to_csv("model_performance_comparison.csv", index=False)
    return df




# 绘制损失和准确率折线图
def plot_training_progress(epoch_data):
    plt.figure(figsize=(12, 6))

    for model_name, data in epoch_data.items():
        epochs = [entry['epoch'] for entry in data]
        loss = [entry['loss'] for entry in data]
        accuracy = [entry['accuracy'] for entry in data]

        # 绘制损失图
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label=f"{model_name} Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.legend()

        # 绘制准确率图
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label=f"{model_name} Accuracy")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epochs')
        plt.legend()

    plt.tight_layout()
    plt.show()

