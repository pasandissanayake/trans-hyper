from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt, _ = build_prompt()

input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
output_ids = model.generate(
    input_ids["input_ids"],
    max_new_tokens=100,
    temperature=0.2
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))