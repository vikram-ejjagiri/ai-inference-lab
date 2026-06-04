import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# MODEL SETUP
# =========================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("=" * 70)
print("LOADING MODEL...")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("\nModel loaded successfully!")

# =========================================================
# BASE TEXT
# =========================================================

base_text = """
Artificial Intelligence is transforming modern technology.
Transformers use attention mechanisms to process language.
Neural networks rely on tensors and matrix multiplication.
GPU acceleration is important for modern AI inference.
"""

# =========================================================
# CREATE DIFFERENT CONTEXT LENGTHS
# =========================================================

prompts = {
    "SMALL": base_text * 5,
    "MEDIUM": base_text * 20,
    "LARGE": base_text * 50,
    "XLARGE": base_text * 100
}

# =========================================================
# RESULTS STORAGE
# =========================================================

results = []

# =========================================================
# BENCHMARK FUNCTION
# =========================================================

def benchmark_prompt(prompt_name, prompt_text):

    print("\n" + "=" * 70)
    print(f"TESTING: {prompt_name}")
    print("=" * 70)

    # -----------------------------------------------------
    # TOKENIZATION
    # -----------------------------------------------------

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt"
    ).to("cuda")

    input_token_count = inputs["input_ids"].shape[1]

    print(f"\nInput Tokens: {input_token_count}")

    # -----------------------------------------------------
    # CLEAR GPU CACHE
    # -----------------------------------------------------

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # -----------------------------------------------------
    # GPU SYNCHRONIZATION
    # IMPORTANT FOR ACCURATE TIMING
    # -----------------------------------------------------

    torch.cuda.synchronize()

    # -----------------------------------------------------
    # START TIMER
    # -----------------------------------------------------

    start_time = time.time()

    # -----------------------------------------------------
    # GENERATE TOKENS
    # -----------------------------------------------------

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        min_new_tokens=50,
        do_sample=False
    )

    # -----------------------------------------------------
    # WAIT FOR GPU TO FINISH
    # -----------------------------------------------------

    torch.cuda.synchronize()

    # -----------------------------------------------------
    # END TIMER
    # -----------------------------------------------------

    end_time = time.time()

    # =========================================================
    # METRICS
    # =========================================================

    generation_time = end_time - start_time

    output_token_count = outputs.shape[1]

    generated_tokens = output_token_count - input_token_count

    tokens_per_second = generated_tokens / generation_time

    gpu_memory = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    results.append({
        "Prompt": prompt_name,
        "Input Tokens": input_token_count,
        "Generated Tokens": generated_tokens,
        "Tokens/sec": tokens_per_second,
        "Time": generation_time,
        "Memory": gpu_memory
    })

    # =========================================================
    # PRINT RESULTS
    # =========================================================

    print(f"\nGenerated Tokens : {generated_tokens}")
    print(f"Generation Time  : {generation_time:.2f} sec")
    print(f"Tokens / Second  : {tokens_per_second:.2f}")
    print(f"GPU Memory Usage : {gpu_memory:.2f} GB")

# =========================================================
# RUN ALL BENCHMARKS
# =========================================================

for name, text in prompts.items():
    benchmark_prompt(name, text)

# =========================================================
# FINAL COMPARISON TABLE
# =========================================================

print("\n" + "=" * 70)
print("FINAL CONTEXT SCALING COMPARISON")
print("=" * 70)

print(
    f"\n{'PROMPT':<12}"
    f"{'INPUT TOKENS':<18}"
    f"{'GEN TOKENS':<15}"
    f"{'TOKENS/SEC':<18}"
    f"{'TIME(sec)':<15}"
    f"{'GPU MEMORY'}"
)

print("-" * 100)

for r in results:

    print(
        f"{r['Prompt']:<12}"
        f"{r['Input Tokens']:<18}"
        f"{r['Generated Tokens']:<15}"
        f"{r['Tokens/sec']:<18.2f}"
        f"{r['Time']:<15.2f}"
        f"{r['Memory']:.2f} GB"
    )

print("\n" + "=" * 70)
print("BENCHMARK COMPLETED")
print("=" * 70)