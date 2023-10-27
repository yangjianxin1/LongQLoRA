from transformers import AutoTokenizer, AutoConfig
import math
from loguru import logger
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append("../../")
from component.utils import ModelUtils
from component.dataset import EvalDataset
from component.collator import EvalCollator
"""
评测模型ppl
"""


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_name_or_path', type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument('--adapter_name_or_path', type=str, default='path-to-lora')
    # parser.add_argument('--eval_file', type=str, default="../../eval/data/proof-pile-test.bin", help='')
    # parser.add_argument('--eval_file', type=str, default="../../eval/data/pg19-test.bin", help='')
    parser.add_argument('--eval_file', type=str, default="../../data/eval/pg19-validation.bin", help='')
    parser.add_argument('--load_in_4bit', type=bool, default=True, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size during inference')
    parser.add_argument('--max_seq_length', type=int, default=7900, help='context length during evaluation')
    parser.add_argument('--model_max_length', type=int, default=8192, help='context size during fine-tuning')
    parser.add_argument('--sliding_window', type=int, default=7900, help='context size during fine-tuning')
    args = parser.parse_args()
    return args


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    acc = n_correct/n_word
    return n_correct, n_word, acc


def main():
    args = parse_args()
    # 修改RoPE的position最大长度
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    logger.info(f'Change model_max_length from {orig_ctx_len} to {args.model_max_length}')

    # 加载模型
    model = ModelUtils.load_model(
        args.model_name_or_path,
        config=config,
        load_in_4bit=args.load_in_4bit,
        adapter_name_or_path=args.adapter_name_or_path
    ).eval()
    print(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # 加载评测数据
    dataset = EvalDataset(args.eval_file, tokenizer, args.max_seq_length, -100, args.sliding_window)
    collator = EvalCollator(tokenizer, args.max_seq_length, ignore_index=-100)
    dataloader = DataLoader(dataset, collate_fn=collator, batch_size=args.batch_size)

    right_token_cnt = 0
    total_token_cnt = 0
    eval_loss = []
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(model.device)
        labels = batch['labels'].to(model.device)
        with torch.no_grad():
            out = model(input_ids, labels=labels)

        eval_loss.append(out.loss.item())
        n_correct, n_word, _ = calculate_acc(out.logits, labels, ignore_index=-100)
        right_token_cnt += n_correct
        total_token_cnt += n_word

    acc = right_token_cnt/total_token_cnt
    eval_loss = sum(eval_loss) / len(eval_loss)
    logger.info(f'Acc: {acc}')
    logger.info(f'Eval loss: {eval_loss}')
    logger.info(f'Eval ppl: {2.71828**eval_loss}')


if __name__ == '__main__':
    main()
