from typing import Any, Dict, List
import torch


class PretrainCollator(object):
    def __init__(self, tokenizer, max_seq_length, ignore_index):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.ignore_index = ignore_index

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
