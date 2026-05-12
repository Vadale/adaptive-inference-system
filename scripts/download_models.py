"""Scarica i modelli HF necessari per Fase 1+ del progetto AIS.

Usage:
  python scripts/download_models.py E2B           # solo router (~10 GB)
  python scripts/download_models.py E4B           # solo decoder  (~16 GB)
  python scripts/download_models.py E2B E4B       # entrambi

Note operative:
  - Repo non gated al 2026-05-11 (`google/gemma-4-E*B-it`). Se in futuro
    diventano gated, il token cached in ~/.cache/huggingface/token basta.
  - `snapshot_download` riprende parziali e fa cache atomico. Safe re-run.
  - Verifica spazio prima: `df -h` — bf16 safetensors monofile, non
    quantizzato (P5: i `gemma4:*` di Ollama sono GGUF Q4_K_M, incompatibili).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_BY_SIZE = {
    "E2B": ("google/gemma-4-E2B-it", 10.3),
    "E4B": ("google/gemma-4-E4B-it", 16.0),
}


def _free_gb(path: str = "/") -> float:
    return shutil.disk_usage(path).free / 1e9


def main(args: list[str]) -> int:
    if not args:
        print(__doc__)
        return 2
    unknown = [a for a in args if a not in REPO_BY_SIZE]
    if unknown:
        print(f"Argomenti sconosciuti: {unknown}. Validi: {list(REPO_BY_SIZE)}")
        return 2

    requested = [(name, *REPO_BY_SIZE[name]) for name in args]
    total_gb = sum(s for _, _, s in requested)
    free_gb = _free_gb()
    print(f"Richiesto: {', '.join(n for n, _, _ in requested)} = {total_gb:.1f} GB")
    print(f"Libero su disco: {free_gb:.1f} GB")
    if free_gb < total_gb + 5:
        print(f"FAIL: meno di 5 GB di margine dopo download. Liberare spazio prima.")
        return 1

    for name, repo_id, size in requested:
        print(f"\n--- Downloading {name} = {repo_id} ({size} GB) ---")
        local = snapshot_download(repo_id=repo_id)
        print(f"  cached at: {local}")

    free_after = _free_gb()
    print(f"\nDone. Disco libero ora: {free_after:.1f} GB "
          f"(consumati ~{free_gb - free_after:.1f} GB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
