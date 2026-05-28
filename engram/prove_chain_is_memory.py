"""
prove_chain_is_memory.py — the experiment that tests the thesis.

Builds a self, then DELETES the entire episode store and reconstructs it by
replaying the chain. If the rebuilt self is identical — same episodes, same
salience, same state_root — then the memory was never in the store. It was in
the chain. The store is just a cache.

Run offline (no model needed — this tests the chain/replay mechanics):
    python -m engram.prove_chain_is_memory --fake
Or for real:
    python -m engram.prove_chain_is_memory
"""

import sys
from .engram import Engram


def fingerprint(memory):
    """A stable, order-independent signature of the live self."""
    return sorted(f"{e.content}|{e.type}|{e.salience:.4f}|{e.dead}"
                  for e in memory.episodes.values())


def run():
    eng = Engram()
    script = [
        "Hi, I'm building a blockchain-based memory system for AI.",
        "My name is Nate and I work mostly in Python.",
        "Let's call the unit of memory a 'trace', not an episode.",
        "I prefer minimal dependencies.",
    ]
    for m in script:
        eng.step(m)

    before_fp = fingerprint(eng.memory)
    before_root = eng.chain._state_root(eng.memory)
    n_episodes = len(eng.memory.episodes)
    print(f"Built a self: {n_episodes} episodes across {len(eng.chain)} blocks.")
    print(f"state_root before wipe: {before_root[:16]}...")

    # --- the destructive test -------------------------------------------
    print("\nDELETING the entire episode store...")
    eng.memory = type(eng.memory)()          # fresh empty store
    assert len(eng.memory.episodes) == 0
    print("  store is now empty:", len(eng.memory.episodes), "episodes.")

    print("\nReplaying the chain from genesis to rebuild the self...")
    eng.rebuild_cache()
    after_fp = fingerprint(eng.memory)
    after_root = eng.chain._state_root(eng.memory)
    print(f"  rebuilt {len(eng.memory.episodes)} episodes.")
    print(f"state_root after rebuild: {after_root[:16]}...")

    # --- the verdict ----------------------------------------------------
    print("\n" + "=" * 60)
    same_self = before_fp == after_fp
    same_root = before_root == after_root
    print("self identical after wipe+replay:", same_self)
    print("state_root matches:              ", same_root)
    print("chain integrity:                 ", eng.verify())
    if same_self and same_root:
        print("\nVERDICT: the chain IS the memory. The store held nothing")
        print("the chain didn't. Identity lives in the blocks.")
    else:
        print("\nVERDICT: mismatch — something about the self lives OUTSIDE")
        print("the chain. The chain is not yet the full source of truth.")
        # show the difference for debugging
        extra_before = set(before_fp) - set(after_fp)
        extra_after = set(after_fp) - set(before_fp)
        if extra_before:
            print("  lost in replay:", list(extra_before)[:3])
        if extra_after:
            print("  appeared in replay:", list(extra_after)[:3])


def run_fake():
    import engram.backend as b
    import engram.consolidate as c
    import engram.engram as e
    import hashlib

    def fake_llm(prompt):
        if "new_episodes" in prompt:
            # deterministic per-message memory so the test is reproducible
            h = hashlib.md5(prompt.encode()).hexdigest()[:6]
            return ('{"new_episodes":[{"content":"memory %s","type":"fact",'
                    '"salience":0.7}],"reinforce":[],"supersede":[]}' % h)
        return "(fake reply)"

    def fake_embed(t):
        # deterministic embedding from text so rebuilt embeddings match
        h = hashlib.md5(t.encode()).digest()
        return [x / 255.0 for x in h[:8]]

    b.llm = fake_llm
    b.embed = fake_embed
    c.llm = fake_llm
    e.llm = fake_llm
    e.embed = fake_embed
    print("Running offline with a deterministic fake backend.\n")
    run()


if __name__ == "__main__":
    if "--fake" in sys.argv:
        run_fake()
    else:
        run()