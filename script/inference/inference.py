from transformers import AutoTokenizer, TextIteratorStreamer, AutoConfig
import torch
from threading import Thread
import math

import sys
sys.path.append("../../")
from component.utils import ModelUtils
"""
适用于base model进行续写任务
"""


def main():
    context_size = 8192
    # 使用合并后的模型进行推理
    model_name_or_path = 'path-to-base-model'
    adapter_name_or_path = None

    # 使用base model和adapter进行推理
    # model_name_or_path = 'path-to-base-model'
    # adapter_name_or_path = 'path-to-lora'

    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    gen_kwargs = {
        'max_new_tokens': 500,
        'top_p': 0.8,
        'temperature': 1.0,
        'repetition_penalty': 1.0,
        'do_sample': True
    }
    # Set RoPE scaling factor
    config = AutoConfig.from_pretrained(model_name_or_path)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)  # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    # 加载模型
    model = ModelUtils.load_model(
        model_name_or_path,
        config=config,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0)

    gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
    gen_kwargs["streamer"] = streamer

    text = input('Input：')
    while True:
        text = text.strip()
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        gen_kwargs["input_ids"] = input_ids
        with torch.no_grad():
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()
            print('Output:')
            for new_text in streamer:
                print(new_text, end='', flush=True)
        print()
        text = input('Input：')


if __name__ == '__main__':
    main()
