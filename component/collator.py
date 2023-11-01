from typing import Any, Dict, List
import torch


class Collator(object):

    def __init__(self, tokenizer, max_seq_length, ignore_index):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch) -> Dict[str, Any]:
        raise ImportError


class PretrainCollator(Collator):
    def __call__(self, batch: List[str]) -> Dict[str, Any]:
        inputs = self.tokenizer(
            batch, return_tensors='pt', max_length=self.max_seq_length,
            truncation=True, padding='max_length'
            # padding=True,
            # add_special_tokens=False
        )
        input_ids = inputs.input_ids
        # 将pad_token_id替换为-100
        labels = torch.where(input_ids != self.tokenizer.pad_token_id, input_ids, self.ignore_index)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': inputs.attention_mask,
            'labels': labels
        }
        return inputs


class EvalCollator(Collator):
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [x['input_ids'] for x in batch]
        labels = [x['labels'] for x in batch]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        inputs = {
            'input_ids': input_ids,
            'labels': labels
        }
        return inputs


class SFTCollator(Collator):

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = []
        attention_mask = []
        labels = []

        for x in batch:
            input_ids.append(x['input_ids'])
            attention_mask.append(x['attention_mask'])
            labels.append(x['labels'])

        # 将list转换为tensor，得到最终的的模型输入
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return inputs
