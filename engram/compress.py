"""
compress.py — forward compression. The distinctive feature.

Deep history doesn't stay perfectly retrievable; it gets summarized forward.
High-salience episodes survive verbatim regardless of age. Old, faint episodes
merge into a single gist episode or tombstone entirely. The cryptographic
skeleton (chain.py) stays exact throughout — only the *content* decays.

This is the brain-like trade: forget the specific dinner, keep 'I like that
restaurant'. A machine implementation, so it's deliberate and tunable rather
than the uncontrolled drift biology was stuck with.

THE INVARIANT: nothing changes the self except by appending a block. Forgetting
is a change to the self, so compression MUST record what it did as a block —
otherwise replaying the chain from genesis would rebuild the un-compressed self
and the chain would quietly stop being the source of truth. Every drop is a
supersede op; every merge is a gist op naming the episodes it absorbed. The
replay module knows how to reconstruct both, so a compressed self survives
replay identically.
"""

from .backend import llm, parse_json

AGE_THRESHOLD = 10        # blocks; older than this is eligible to decay
KEEP_SALIENCE = 0.7       # at/above this, survive verbatim no matter how old
DROP_SALIENCE = 0.15      # below this and old -> tombstone (survives as hash only)

GIST_PROMPT = """Merge these old, low-importance memories into ONE concise gist memory
that preserves their overall meaning. Return STRICT JSON, no prose:
{{"new_episodes": [{{"content": "the merged gist", "type": "insight", "salience": 0.4}}]}}

Memories to merge:
{memories}"""


def compress(memory, chain, embed) -> dict:
    """Run one compression pass and COMMIT it as a block.

    Builds a delta describing every change (drops + merge), applies it to the
    live self, then appends one block recording that delta. After this, the
    self can still be rebuilt exactly by replaying the chain.
    """
    current = len(chain)
    report = {"kept": 0, "dropped": 0, "merged": 0, "gist_created": 0}
    delta = []

    candidates = []
    for e in memory.live():
        age = current - e.last_seen_block
        if age < AGE_THRESHOLD or e.salience >= KEEP_SALIENCE or e.gist:
            report["kept"] += 1
            continue
        if e.salience < DROP_SALIENCE:
            memory.supersede(e.id)            # tombstone; hash persists in its block
            delta.append({"op": "supersede", "id": e.id, "reason": "decayed"})
            report["dropped"] += 1
        else:
            candidates.append(e)              # mid-salience old memories -> merge to gist

    # merge the remaining old-but-not-trivial memories into a single gist
    if len(candidates) >= 2:
        listing = "\n".join(f"- {e.content}" for e in candidates)
        out = parse_json(llm(GIST_PROMPT.format(memories=listing)))
        gists = out.get("new_episodes", [])
        if gists and gists[0].get("content", "").strip():
            g = gists[0]
            content = g["content"].strip()
            ge = memory.add(
                content=content,
                etype="insight",
                salience=float(g.get("salience", 0.4)),
                embedding=embed(content),
                block_index=current,
            )
            ge.gist = True
            merged_ids = [e.id for e in candidates]
            for e in candidates:
                memory.supersede(e.id)
            # one gist op carries: the new gist episode AND the ids it absorbed
            delta.append({
                "op": "gist",
                "id": ge.id,
                "content": content,
                "salience": ge.salience,
                "merged": merged_ids,
            })
            report["merged"] = len(candidates)
            report["gist_created"] = 1

    # commit the forgetting as a block — only if something actually changed
    if delta:
        chain.add_block(f"[compression pass at block {current}]", delta, memory)

    return report