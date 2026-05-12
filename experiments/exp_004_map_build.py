"""exp_004 — Build/save/load TopologicalMap dal NPZ Fase 1."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.topological_map import TopologicalMap, MapEntry

CORPUS_NPZ = ROOT / "corpus" / "activations_gemma_e2b_n5000_L9_last.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-decoder-layers", type=int, default=42,
                    help="42 per E4B, 35 per E2B")
    ap.add_argument("--map-dir", type=str, default=None,
                    help="default: mappa/topology (E4B) o mappa/topology_e2b")
    args = ap.parse_args()
    N_DECODER_LAYERS = args.n_decoder_layers
    if args.map_dir:
        MAP_DIR = Path(args.map_dir)
    elif N_DECODER_LAYERS == 35:
        MAP_DIR = ROOT / "mappa" / "topology_e2b"
    else:
        MAP_DIR = ROOT / "mappa" / "topology"
    print(f"Loading corpus from {CORPUS_NPZ}...")
    data = np.load(CORPUS_NPZ, allow_pickle=True)
    embeddings = data["embeddings"]
    categories = data["categories"]
    meta = data["meta"].item()
    print(f"  N={len(embeddings)} hidden={embeddings.shape[1]} "
          f"pivot=L{meta['pivot_layer']} pool={meta['pool']}")

    print(f"\nBuilding TopologicalMap...")
    m = TopologicalMap(
        hidden_dim=embeddings.shape[1],
        n_decoder_layers=N_DECODER_LAYERS,
    )
    entries = [MapEntry(domain=str(c)) for c in categories]
    m.add_batch(embeddings, entries)
    print(f"  added {len(m)} entries")

    print(f"\nSaving to {MAP_DIR}...")
    m.save(MAP_DIR)
    on_disk = sum(p.stat().st_size for p in MAP_DIR.iterdir()) / 1e6
    print(f"  on disk: {on_disk:.1f} MB")
    for p in sorted(MAP_DIR.iterdir()):
        print(f"    {p.name:20s} {p.stat().st_size/1e6:6.2f} MB")

    # --- Reload + sanity ---
    print(f"\nReload + sanity...")
    m2 = TopologicalMap.load(MAP_DIR)
    assert len(m2) == len(m), "n_entries mismatch dopo load"
    assert m2.hidden_dim == m.hidden_dim
    assert m2.n_decoder_layers == m.n_decoder_layers

    # Query 1: embedding noto (#0) → deve trovare se stesso top-1 con sim ≈ 1.0
    q0 = embeddings[0]
    top5 = m2.lookup(q0, k=5)
    print(f"\n  Query: embedding #0 (categoria '{categories[0]}')")
    for sim, idx, entry in top5:
        marker = " ← self" if idx == 0 else ""
        print(f"    sim={sim:.4f}  idx={idx:4d}  domain={entry.domain}{marker}")
    sim_self, idx_self, _ = top5[0]
    assert idx_self == 0, f"top-1 non è il self! idx={idx_self}"
    assert sim_self > 0.999, f"self similarity {sim_self} < 0.999"

    # Query 2: centroid di una categoria → top-k devono essere principalmente
    # della stessa categoria (sanity check sulla qualità della mappa).
    print("\n  Per-category centroid retrieval (top-10):")
    cats_unique = sorted(set(map(str, categories)))
    pass_centroid = 0
    for c in cats_unique:
        mask = categories == c
        centroid = embeddings[mask].mean(axis=0)
        top10 = m2.lookup(centroid, k=10)
        same_cat = sum(1 for _, _, e in top10 if e.domain == c)
        ok = same_cat >= 7  # ≥ 70%
        pass_centroid += int(ok)
        marker = "[OK]" if ok else "[FAIL]"
        print(f"    {c:22s} same-category in top-10: {same_cat}/10  {marker}")

    print("\n" + "=" * 60)
    print(f"  Self-retrieve: PASS (sim={sim_self:.4f}, idx=0)")
    print(f"  Centroid retrieval ≥7/10 same-category: {pass_centroid}/{len(cats_unique)} categorie")
    rc = 0 if pass_centroid == len(cats_unique) else 1
    print(f"  TopologicalMap smoke: {'PASS' if rc == 0 else 'PARTIAL'}")
    print("=" * 60)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
