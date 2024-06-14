from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import sys
sys.path.append("../")
from llm_eval.proxy.services.chat_instruction_completor import LlamaChatCompleter, MetaChatCompleter
def get_eos_token_id(tokenizer):
    return [tokenizer.encode(eos_token)[0] for eos_token in ["USER", "ASSISTANT", "<|endoftext|>"]]
model_path = '/home/dpf/workspace/huggingface/llm/WizardLM/WizardLM-7B-V1.0'
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
generation_config = GenerationConfig.from_pretrained(model_path)
# eos_token_id = get_eos_token_id(tokenizer)
# print(eos_token_id, tokenizer.eos_token_id)
# generation_config.eos_token_id = eos_token_id
model.generation_config.update(**generation_config.to_diff_dict())
print(generation_config.eos_token_id)
query = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: 你好，你知道太阳从什么方向升起吗? ASSISITANT:"
# prompt = LlamaChatCompleter().complete_chat_instruction(query,[])
# print(prompt)

input_ids = tokenizer.encode(query, return_tensors='pt')
output_ids = model.generate(input_ids = input_ids.to(model.device), max_length = 2048)[0]
outputs = tokenizer.decode(output_ids)
print(outputs)

