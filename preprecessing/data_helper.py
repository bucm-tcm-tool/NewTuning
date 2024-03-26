from torch.utils.data import Dataset
import json
from tqdm import tqdm
import transformers
import random
import torch

tokenizer = transformers.AutoTokenizer.from_pretrained(
    '/root/Documents/mChatAdapter/chatglm2', trust_remote_code=True)

class TextLoader(Dataset):

    def __init__(self, base_dir):
        # base_dir: /root/data/CMeEE/
        # data_name: CMeEE
        with open(base_dir, 'r', encoding='utf-8') as fp:
            self.data = json.load(fp)
            #random.seed(12345)
            #random.shuffle(self.data)
            #half_data = int(len(self.data)*0.25)
            #self.data = self.data[:half_data]

            # self.max_seq_length = max_seq_length
        # self.skip_over_length = skip_over_length

    def __len__(self):
        # 50 75000
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return sample

def collate_fn(samples):

    len_ids = [len(feature["input_ids"]) for feature in samples]
    longest = max(len_ids)  # 之后按照batch中最长的input_ids进行padding

    input_ids = []
    labels_list = []

    for length, sample in sorted(zip(len_ids, samples), key=lambda x: -x[0]):
        if 'train' == sample['ds_type']:
            ids = sample["input_ids"]
            context_len = sample["context_len"]

            labels = (
                    [-100] * (context_len - 1) + ids[(context_len - 1):] + [-100] * (longest - length)
            )  # -100标志位后面会在计算loss时会被忽略不贡献损失，我们集中优化target部分生成的loss

            ids = ids + [tokenizer.pad_token_id] * (longest - length)

            input_ids.append(torch.LongTensor(ids))
            labels_list.append(torch.LongTensor(labels))
        else:
            context_len = sample["context_len"]

            ids = sample["input_ids"][:context_len]
            ids = ids + [tokenizer.pad_token_id] * (longest - len(ids))
            input_ids.append(torch.LongTensor(ids))

            labels = sample["input_ids"][context_len:]
            labels = labels+ [-100] * (longest - len(labels))

            labels_list.append(torch.LongTensor(labels))

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }
