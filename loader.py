import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer



class DataGenerator(Dataset):
    """
    数据生成器，继承自torch.utils.data.Dataset。

    该类负责从指定路径加载数据，并对数据进行预处理，包括分词、编码等操作。
    它还定义了如何获取数据集中的条目和数据集的长度。

    属性:
    - data_path: 数据文件的路径。
    - config: 包含各种配置参数的字典，如预训练模型路径、最大长度等。
    - idx2label: 将标签索引映射到标签名称的字典。
    - label2idx: 将标签名称映射到标签索引的字典。
    - tokenizer: BertTokenizer实例，用于文本的分词和编码。
    - vocab: 词汇表，将单词映射到其在词汇表中的索引。
    """

    def __init__(self, data_path, config):
        """
        初始化数据生成器。

        参数:
        - data_path: 数据文件的路径。
        - config: 包含各种配置参数的字典。
        """
        self.data_path = os.path.abspath(data_path)
        self.config = config
        self.idx2label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
                          5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
                          10: '体育', 11: '科技', 12: '汽车', 13: '健康',
                          14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        self.label2idx = {v: k for k, v in self.idx2label.items()}
        self.tokenizer = BertTokenizer.from_pretrained(config['pretrained_path'])
        self.vocab = {}
        self.loader_vocab()
        self.loader()
        self.config["vocab_size"] = len(self.vocab)

    def __len__(self):
        """
        获取数据集的长度。

        返回:
        - 数据集的长度。
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        获取数据集中的指定条目。

        参数:
        - index: 条目的索引。

        返回:
        - input_id: 编码后的文本。
        - label: 文本对应的标签。
        """
        input_id, label = self.data[index]
        return input_id, label

    def loader(self):
        """
        从文件中加载数据，并对每条数据进行预处理。
        包括将文本转换为编码、将标签转换为索引，并将它们存储在self.data中。
        """
        self.data = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        line_data = json.loads(line.strip())
                        tag = line_data.get('tag')
                        if tag not in self.label2idx:
                            continue
                        label = self.label2idx[tag]
                        title = line_data.get('title', '')
                        if not title:
                            continue
                        input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                         pad_to_max_length=True)
                        label_tensor = torch.LongTensor([label])
                        input_id_tensor = torch.LongTensor(input_id)
                        self.data.append([input_id_tensor, label_tensor])
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"File not found: {self.data_path}")
        except Exception as e:
            print(f"An error occurred while loading data: {e}")

    def loader_vocab(self):
        """
        从文件中加载词汇表，并将其存储在self.vocab中。
        每个单词分配一个唯一的索引。
        """
        vocab_path = os.path.abspath(self.config['vocab_path'])
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.vocab[word] = len(self.vocab) + 1
        except FileNotFoundError:
            print(f"Vocabulary file not found: {vocab_path}")
        except Exception as e:
            print(f"An error occurred while loading vocabulary: {e}")


def load_data(data_path, config):
    """
    创建并返回一个数据加载器。

    参数:
    - data_path: 数据文件的路径。
    - config: 包含各种配置参数的字典。

    返回:
    - dataloader: DataLoader实例，用于迭代数据集。
    """
    dataset = DataGenerator(data_path, config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    return dataloader
