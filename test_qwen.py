from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2-7B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",         # 用新名字，避免 warning
    device_map="auto"     # 需要 accelerate
)

def chat_once(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0
        )

    # 只取新生成的部分
    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip()

if __name__ == "__main__":
    ans = chat_once("简单介绍一下 CURE-Bench 这个比赛。")
    print("模型输出：")
    print(ans)
