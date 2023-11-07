from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LongQLoRAArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    model_max_length: int = field(metadata={"help": "模型位置编码扩展为该长度"})
    train_file: str = field(metadata={"help": "训练数据路径"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    sft: bool = field(metadata={"help": "True为sft，False则进行自回归训练"})

    target_modules: str = field(default=None, metadata={
        "help": "QLoRA插入adapter的位置，以英文逗号分隔。如果为None，则在自动搜索所有linear，并插入adapter"
    })
    eval_file: str = field(default=None, metadata={"help": "评测集路径"})
    use_flash_attn: bool = field(default=False, metadata={"help": "训练时是否使用flash attention"})
    train_embedding: bool = field(default=False, metadata={"help": "词表权重是否参与训练"})
    train_norm: bool = field(default=False, metadata={"help": "norm权重是否参与训练"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})


