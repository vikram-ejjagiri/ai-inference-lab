import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Small lightweight model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded successfully!")

prompt = """
<|system|>
You are a helpful AI assistant.
<|user|>
Explain quantization in simple words.
<|assistant|>
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating response...")

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== MODEL RESPONSE ===\n")
print(response)