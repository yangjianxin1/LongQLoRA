from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import os
"""
使用该脚本，将lora的权重合并到base model中
"""


def merge_lora_to_base_model():
    model_name_or_path = 'NousResearch/Llama-2-7b-hf'
    adapter_name_or_path = 'path-to-lora-8k'
    save_path = '../checkpoint/llama2-7b-longqlora-8k'

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    # 加载base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )
    # 更新base model的部分权重
    trainable_params_file = os.path.join(adapter_name_or_path, "trainable_params.bin")
    if os.path.isfile(trainable_params_file):
        model.load_state_dict(torch.load(trainable_params_file, map_location=model.device), strict=False)
    # 合并lora权重
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()
