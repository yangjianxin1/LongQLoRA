from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoConfig
)
import argparse
from loguru import logger
import os
from os.path import join
import yaml
import torch
import bitsandbytes as bnb
import math
from collections import defaultdict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from component.collator import PretrainCollator
from component.dataset import PretrainDataset
from component.argument import LongQLoRAArguments
from component.trainer import LoRATrainer
from component.loss import CausalLMLoss
from attention.llama_attn_replace import replace_llama_attn


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)

    # 查看参与训练的参数情况
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total model params: %.2fM" % (total / 1e6))
    logger.info(
        f'trainable params: {trainable} || all params: {total} || trainable%: {round(trainable / total, 4)}')


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='./train_args/llama2-7b-pretrain.yaml', help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((LongQLoRAArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_yaml_file(yaml_file=train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    logger.info("train_args:{}".format(training_args))
    # 加载训练配置文件
    with open(train_args_file, "r") as f:
        train_args = yaml.safe_load(f)
    # 保存训练参数到输出目录
    with open(join(training_args.output_dir, 'train_args.yaml'), "w") as f:
        yaml.dump(train_args, f)
    # 设置随机种子
    set_seed(training_args.seed)
    training_args.train_embedding = args.train_embedding
    return args, training_args


def load_model_and_tokenizer(args, training_args):
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config.use_cache = False
    model_type = config.model_type
    assert model_type == 'llama', "Only support llama and gpt-neox for now"
    replace_llama_attn(args.use_flash_attn)

    # 修改RoPE的position最大长度
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    logger.info(f'Change model_max_length from {orig_ctx_len} to {getattr(config, "max_position_embeddings")}')

    # 设置device_map，以适配多卡训练
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}

    # 加载模型
    logger.info(f'Loading model from: {args.model_name_or_path}')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        device_map=device_map,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        ),
    )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        # use_fast=True,
        use_fast=False if config.model_type == 'llama' else True
    )

    assert tokenizer.eos_token_id is not None
    assert tokenizer.bos_token_id is not None
    # 部分tokenizer的pad_token_id为None
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    # casts all the non int8 modules to full precision (fp32) for stability
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    print(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
    return model, tokenizer


def insert_adapter(args, model):
    # 找到所有需要插入adapter的位置
    if args.target_modules is not None:
        target_modules = args.target_modules.split(',')
    else:
        target_modules = find_all_linear_names(model)
    # 初始化lora配置
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None
        # modules_to_save=["embed_tokens", "lm_head"] if args.train_embedding else None
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.config.torch_dtype = torch.float32

    # 根据配置，决定word embedding和norm是否参与训练
    for n, p in model.named_parameters():
        # 训练word embedding
        if args.train_embedding and ("embed_tokens" in n or "lm_head" in n):
            p.requires_grad = True
        # 训练norm
        if args.train_norm and "norm" in n:
            p.requires_grad = True

    # 查看模型种各种类型的参数的情况
    verify_model_dtype(model)

    return model


def merge_lora():
    pass


def init_components(args, training_args):
    """
    初始化各个组件
    """
    logger.info('Initializing components...')
    # 务必设为False，否则多卡训练会报错
    training_args.ddp_find_unused_parameters = False
    # 加载model和tokenizer
    model, tokenizer = load_model_and_tokenizer(args, training_args)
    # 插入adapter
    model = insert_adapter(args, model)
    # 初始化损失函数
    loss_func = CausalLMLoss(ignore_index=-100)

    # 加载训练集和验证集
    train_dataset = PretrainDataset(args.train_file, tokenizer, args.max_seq_length)
    eval_dataset = None if args.eval_file is None else PretrainDataset(args.eval_file, tokenizer, args.max_seq_length)
    data_collator = PretrainCollator(tokenizer, args.max_seq_length, -100)

    # 初始化Trainer
    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_loss=loss_func
    )
    return trainer


def main():
    # 进行一些配置和检查
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = init_components(args, training_args)
    # 开始训练
    logger.info("*** starting training ***")
    train_result = trainer.train()
    # 保存最后的checkpoint
    trainer.save_model(training_args.output_dir)  # Save the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
