"""
inspect_retrieval.py — see what the self knows and what it surfaced.

When a real reply misses an answer the self should have, run this against the
persisted chain to see (a) every live memory, (b) its similarity to the query,
and (c) whether it's in the top fraction guaranteed to surface.

    python -m engram.inspect_retrieval "what's my name"
    python -m engram.inspect_retrieval "what's my name" engram_portable_self.json
"""

import sys
from .engram import Engram
from .backend import embed
from .memory import cosine, MemoryStore


def inspect(query: str, path: str = "engram_portable_self.json"):
    eng = Engram.load(path)
    print(f"loaded {len(eng.memory.live())} live memories from {len(eng.chain)} blocks\n")

    q = embed(query)
    rows = []
    for e in eng.memory.live():
        sim = cosine(q, e.embedding)
        recency = 1.0 / (1 + len(eng.chain) - e.last_seen_block)
        blended = (MemoryStore.W_SIM * sim
                   + MemoryStore.W_SAL * e.salience
                   + MemoryStore.W_REC * recency)
        rows.append((sim, blended, e))
    rows.sort(key=lambda r: r[1], reverse=True)

    # compute the relative cutoff (matches MemoryStore.retrieve)
    sims_desc = sorted((s for s, _, _ in rows), reverse=True)
    cutoff_idx = max(0, int(len(sims_desc) * MemoryStore.SIM_TOP_FRACTION) - 1)
    sim_cutoff = sims_desc[cutoff_idx] if sims_desc else 1.0

    print(f"query: {query!r}")
    print(f"guaranteed-surface cutoff (top {MemoryStore.SIM_TOP_FRACTION*100:.0f}%): "
          f"sim >= {sim_cutoff:.3f}\n")
    print(f"{'sim':>6}  {'blend':>6}  top?   content")
    print("-" * 70)
    for sim, blended, e in rows:
        flag = "YES" if sim >= sim_cutoff else "  ."
        print(f"{sim:>6.3f}  {blended:>6.3f}   {flag}    [{e.salience:.2f}] {e.content}")

    chosen = eng.memory.retrieve(q, len(eng.chain), k=8)
    print(f"\nretrieve(k=8) would return these {len(chosen)} memories:")
    for e in chosen:
        print(f"  - {e.content}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    query = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "engram_portable_self.json"
    inspect(query, path)