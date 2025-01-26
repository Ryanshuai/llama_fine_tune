from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer


model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)


