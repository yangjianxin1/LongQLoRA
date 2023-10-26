from loguru import logger
import json
from torch.utils.data import Dataset
import numpy as np


class PretrainDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length, ignore_index=-100):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()

        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        text = json.loads(data)['text']
        return text


class EvalDataset(Dataset):
    """
    用于评测ppl
    """
    def __init__(self, file, tokenizer, max_seq_length, ignore_index=-100, sliding_window=256):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.pad_token_id = tokenizer.pad_token_id
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        token_list = np.memmap(file, dtype=np.uint16, mode='r').tolist()

        # 以滑动窗口截取评测数据
        eval_data_list = []
        for i in range(0, len(token_list), sliding_window):
            input_ids = token_list[i: i+max_seq_length]
            labels = token_list[i: i+max_seq_length]
            # padding
            padding_len = self.max_seq_length - len(input_ids)
            input_ids += [self.pad_token_id]*padding_len
            labels += [self.ignore_index]*padding_len
            eval_data_list.append({
                'input_ids': input_ids,
                'labels': labels
            })
        logger.info("there are {} data in eval dataset".format(len(eval_data_list)))
        self.data_list = eval_data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        return data
