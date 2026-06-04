import time
import gc
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt = """
<|system|>
You are a helpful AI assistant.
<|user|>
Explain quantization in simple words.
<|assistant|>
"""

# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

print("Loading tokenizer...\n")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ==================================================
# FP16 MODEL
# ==================================================

print("======================================")
print("LOADING FP16 MODEL")
print("======================================\n")

start_time = time.time()

fp16_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

fp16_load_time = time.time() - start_time

# --------------------------------------------------
# GPU MEMORY USAGE
# --------------------------------------------------

if torch.cuda.is_available():
    fp16_memory = torch.cuda.memory_allocated() / 1024**3
else:
    fp16_memory = 0

print(f"FP16 GPU Memory Usage: {fp16_memory:.2f} GB")

# --------------------------------------------------
# TOKENIZE INPUT
# --------------------------------------------------

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(fp16_model.device)

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------

print("\nRunning FP16 inference...\n")

start_time = time.time()

outputs = fp16_model.generate(
    **inputs,
    max_new_tokens=100
)

fp16_inference_time = time.time() - start_time

# --------------------------------------------------
# DECODE OUTPUT
# --------------------------------------------------

fp16_response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\n========== FP16 RESPONSE ==========\n")
print(fp16_response)

print("\n========== FP16 STATS ==========")
print(f"FP16 Load Time: {fp16_load_time:.2f} sec")
print(f"FP16 Inference Time: {fp16_inference_time:.2f} sec")
print(f"FP16 GPU Memory: {fp16_memory:.2f} GB")

# ==================================================
# CLEAN GPU MEMORY
# ==================================================

print("\nCleaning GPU memory...\n")

del fp16_model
del outputs

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==================================================
# INT8 QUANTIZED MODEL
# ==================================================

print("======================================")
print("LOADING INT8 MODEL")
print("======================================\n")

quant_config = BitsAndBytesConfig(
    load_in_8bit=True
)

start_time = time.time()

int8_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

int8_load_time = time.time() - start_time

# --------------------------------------------------
# GPU MEMORY USAGE
# --------------------------------------------------

if torch.cuda.is_available():
    int8_memory = torch.cuda.memory_allocated() / 1024**3
else:
    int8_memory = 0

print(f"INT8 GPU Memory Usage: {int8_memory:.2f} GB")

# --------------------------------------------------
# TOKENIZE INPUT
# --------------------------------------------------

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(int8_model.device)

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------

print("\nRunning INT8 inference...\n")

start_time = time.time()

outputs = int8_model.generate(
    **inputs,
    max_new_tokens=100
)

int8_inference_time = time.time() - start_time

# --------------------------------------------------
# DECODE OUTPUT
# --------------------------------------------------

int8_response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\n========== INT8 RESPONSE ==========\n")
print(int8_response)

print("\n========== INT8 STATS ==========")
print(f"INT8 Load Time: {int8_load_time:.2f} sec")
print(f"INT8 Inference Time: {int8_inference_time:.2f} sec")
print(f"INT8 GPU Memory: {int8_memory:.2f} GB")

# ==================================================
# FINAL COMPARISON
# ==================================================

print("\n======================================")
print("FINAL COMPARISON")
print("======================================\n")

print(f"FP16 Inference Time : {fp16_inference_time:.2f} sec")
print(f"INT8 Inference Time : {int8_inference_time:.2f} sec\n")

print(f"FP16 GPU Memory : {fp16_memory:.2f} GB")
print(f"INT8 GPU Memory : {int8_memory:.2f} GB")

print("\nExperiment completed successfully!")