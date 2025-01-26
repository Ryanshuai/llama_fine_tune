from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "unsloth/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

lora_rank = 8
lora_alpha = 32
lora_dropout = 0.1

lora_config = LoraConfig(
    r=lora_rank,  # LoRA 秩
    lora_alpha=lora_alpha,  # LoRA alpha参数
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 需要训练的模块
    lora_dropout=lora_dropout,  # LoRA dropout
    bias="none",  # 是否训练偏置项
    task_type=TaskType.CAUSAL_LM  # 任务类型
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数比例
