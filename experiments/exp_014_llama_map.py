"""exp_014 — build TopologicalMap on Llama 3.2 3B corpus (28 layers)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.topological_map import TopologicalMap, MapEntry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-npz", type=str, default=None,
                    help="default: corpus/activations_llama32_3b_n5000_L9_last.npz")
    ap.add_argument("--n-decoder-layers", type=int, default=28,
                    help="28 per Llama 3.2 3B")
    ap.add_argument("--map-dir", type=str, default=None,
                    help="default: mappa/topology_llama32_3b")
    args = ap.parse_args()

    corpus_npz = Path(args.corpus_npz) if args.corpus_npz else (
        ROOT / "corpus" / "activations_llama32_3b_n5000_L9_last.npz"
    )
    map_dir = Path(args.map_dir) if args.map_dir else (
        ROOT / "mappa" / "topology_llama32_3b"
    )

    print(f"Loading corpus from {corpus_npz}...", flush=True)
    data = np.load(corpus_npz, allow_pickle=True)
    embeddings = data["embeddings"]
    categories = data["categories"]
    meta = data["meta"].item()
    print(f"  N={len(embeddings)} hidden={embeddings.shape[1]} "
          f"pivot=L{meta['pivot_layer']} pool={meta['pool']}", flush=True)

    print(f"\nBuilding TopologicalMap (n_decoder_layers={args.n_decoder_layers})...",
          flush=True)
    m = TopologicalMap(
        hidden_dim=embeddings.shape[1],
        n_decoder_layers=args.n_decoder_layers,
    )
    entries = [MapEntry(domain=str(c)) for c in categories]
    m.add_batch(embeddings, entries)
    print(f"  added {len(m)} entries", flush=True)

    print(f"\nSaving to {map_dir}...", flush=True)
    m.save(map_dir)
    on_disk = sum(p.stat().st_size for p in map_dir.iterdir()) / 1e6
    print(f"  on disk: {on_disk:.1f} MB", flush=True)

    print("\nReload + sanity...", flush=True)
    m2 = TopologicalMap.load(map_dir)
    assert len(m2) == len(m)
    q0 = embeddings[0]
    top5 = m2.lookup(q0, k=5)
    sim_self, idx_self, _ = top5[0]
    assert idx_self == 0 and sim_self > 0.999

    cats_unique = sorted(set(map(str, categories)))
    pass_centroid = 0
    print("\n  Per-category centroid retrieval (top-10):", flush=True)
    for c in cats_unique:
        mask = categories == c
        centroid = embeddings[mask].mean(axis=0)
        top10 = m2.lookup(centroid, k=10)
        same_cat = sum(1 for _, _, e in top10 if e.domain == c)
        ok = same_cat >= 7
        pass_centroid += int(ok)
        marker = "[OK]" if ok else "[FAIL]"
        print(f"    {c:22s} {same_cat}/10  {marker}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"  Self-retrieve: PASS (sim={sim_self:.4f})", flush=True)
    print(f"  Centroid ≥7/10: {pass_centroid}/{len(cats_unique)}", flush=True)
    rc = 0 if pass_centroid == len(cats_unique) else 1
    print(f"  Llama mappa: {'PASS' if rc == 0 else 'PARTIAL'}", flush=True)
    print("=" * 60, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
