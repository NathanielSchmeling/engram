"""
demo_compression.py — watch the self forget, and prove the forgetting is real.

Builds a longer history so early low-salience memories age past the threshold,
runs a compression pass, and shows three things:
  - the self forgot: old faint memories merged into gist or dropped
  - the skeleton stayed exact: chain still verifies
  - the forgetting is on the chain: wipe + replay rebuilds the COMPRESSED self

    python -m engram.demo_compression --fake
    python -m engram.demo_compression
"""

import sys
from .engram import Engram


def fp(mem):
    return sorted(f"{e.content}|{e.salience:.4f}|{e.dead}|{e.gist}"
                  for e in mem.episodes.values())


def run():
    eng = Engram()

    # a longer life so early memories age past AGE_THRESHOLD (10 blocks)
    script = [
        "Hi, I'm building a blockchain memory system for AI.",
        "Random aside: I had a sandwich for lunch today.",
        "My name is Sam.",
        "The weather is cloudy where I am.",
        "I work mostly in Python.",
        "I mentioned earlier I like coffee in the mornings.",
        "Let's call the unit of memory a 'trace'.",
        "By the way my cat is named Pixel.",
        "I prefer minimal dependencies.",
        "I once used Java but didn't enjoy it.",
        "The core idea is that the chain IS the memory.",
        "I'm based somewhere with a temperate climate.",
        "Persistence is now working in the project.",
    ]
    for m in script:
        eng.step(m)

    print(f"Lived {len(eng.chain)} blocks, {len(eng.memory.live())} live memories.\n")
    print("Before compression — the full self:")
    for e in eng.memory.live():
        print(f"  [{e.salience:.2f}] {e.content}")

    rep = eng.compress()
    print(f"\nCompression pass: {rep}")

    print("\nAfter compression — what survived:")
    for e in eng.memory.live():
        tag = "GIST" if e.gist else f"{e.salience:.2f}"
        print(f"  [{tag}] {e.content}")

    # prove the forgetting is on the chain
    before, before_root = fp(eng.memory), eng.chain._state_root(eng.memory)
    eng.rebuild_cache()
    after, after_root = fp(eng.memory), eng.chain._state_root(eng.memory)

    print("\n" + "=" * 60)
    print("compressed self survives wipe+replay:", before == after and before_root == after_root)
    print("chain integrity:                     ", eng.verify())
    print("\nThe self forgot like a mind — and the forgetting itself is")
    print("recorded on the chain, so the chain remains the whole truth.")


def run_fake():
    import engram.backend as b, engram.consolidate as c
    import engram.engram as e, engram.compress as cm
    import hashlib

    n = [0]
    def fl(p):
        if "Merge these old" in p:
            return ('{"new_episodes":[{"content":"early misc chatter (lunch, '
                    'weather, cat, java)","type":"insight","salience":0.4}]}')
        if "new_episodes" in p:
            n[0] += 1
            sal = 0.2 if n[0] % 2 == 0 else 0.8
            return ('{"new_episodes":[{"content":"memory %d","type":"fact",'
                    '"salience":%.2f}],"reinforce":[],"supersede":[]}' % (n[0], sal))
        return "(fake reply)"
    def fe(t):
        return [x/255.0 for x in hashlib.md5(t.encode()).digest()[:8]]

    b.llm=fl; b.embed=fe; c.llm=fl; e.llm=fl; e.embed=fe; cm.llm=fl
    print("Running offline with a deterministic fake backend.\n")
    run()


if __name__ == "__main__":
    if "--fake" in sys.argv:
        run_fake()
    else:
        run()