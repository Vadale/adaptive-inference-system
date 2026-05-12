"""Cross-check rapido: GPT-2 Small su 'The capital of France is' in fp32 vs bf16.

Se fp32 predice ' Paris' e bf16 ' the': è precision loss di bf16 — atteso su
124M params, importante saperlo per Gemma 4 (E2B ~5B in bf16 lo regge meglio).
Se entrambi predicono ' the': GPT-2 Small (124M) semplicemente non sa il fatto
con sufficiente confidenza, e il CLAUDE.md va corretto.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT = "The capital of France is"
MODEL_ID = "gpt2"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
inputs = tok(PROMPT, return_tensors="pt")

for label, dtype, device in [
    ("fp32 / cpu", torch.float32, "cpu"),
    ("bf16 / mps", torch.bfloat16, "mps"),
]:
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    m.eval()
    with torch.no_grad():
        logits = m(**{k: v.to(device) for k, v in inputs.items()}).logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)
    top5 = probs.topk(5)
    print(f"\n--- {label} ---")
    for p, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  {tok.decode(idx)!r}  p={p:.4f}")
    paris_id = tok.encode(" Paris", add_special_tokens=False)[0]
    print(f"  rank of ' Paris' (id={paris_id}): {(probs.argsort(descending=True) == paris_id).nonzero().item() + 1} / {len(probs)}")
    print(f"  prob(' Paris') = {probs[paris_id].item():.4f}")
    del m
