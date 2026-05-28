"""
prove_self_survives_restart.py — continuity across sessions.

Build a self, save it, destroy the entire Engram object (simulating shutdown),
then wake a fresh one from disk. If the state_root matches, the self that boots
is the same self that shut down — and it was carried entirely by the chain,
because the chain is all we saved.

This is the persistence analogue of prove_chain_is_memory: that test wiped the
store in memory; this one wipes the whole process.

    python -m engram.prove_self_survives_restart --fake
    python -m engram.prove_self_survives_restart
"""

import os
import sys
from .engram import Engram

PATH = "engram_test_chain.json"


def fingerprint(memory):
    return sorted(f"{e.content}|{e.type}|{e.salience:.4f}|{e.dead}"
                  for e in memory.episodes.values())


def run():
    # --- session 1: live a little, then save and "shut down" ------------
    eng = Engram()
    for m in [
        "Hi, I'm building a blockchain-based memory system for AI.",
        "My name is Sam and I work mostly in Python.",
        "Let's call the unit of memory a 'trace'.",
    ]:
        eng.step(m)

    before_fp = fingerprint(eng.memory)
    before_root = eng.chain._state_root(eng.memory)
    print(f"Session 1: built a self with {len(eng.memory.episodes)} episodes.")
    print(f"  state_root: {before_root[:16]}...")
    eng.save(PATH)
    print(f"  saved chain to {PATH} ({len(eng.chain)} blocks).")

    # destroy everything in memory — simulate closing the program
    del eng
    print("\n...process shut down. Engram object destroyed...\n")

    # --- session 2: wake from disk -------------------------------------
    eng2 = Engram.load(PATH)
    after_fp = fingerprint(eng2.memory)
    after_root = eng2.chain._state_root(eng2.memory)
    print(f"Session 2: woke a self from disk with {len(eng2.memory.episodes)} episodes.")
    print(f"  state_root: {after_root[:16]}...")
    print(f"  integrity:  {eng2.verify()}")

    # --- verdict --------------------------------------------------------
    print("\n" + "=" * 60)
    same = (before_fp == after_fp) and (before_root == after_root)
    if same:
        print("VERDICT: same self across restart. Continuity survived shutdown,")
        print("carried entirely by the chain. The self has a life across sessions.")
    else:
        print("VERDICT: the woken self differs from the one that shut down.")

    # the self can now keep living — prove it accepts new experience
    if same:
        reply = eng2.step("What do I call the unit of memory?")
        print(f"\nThe revived self, asked what it calls memory:\n  {reply[:200]}")

    # cleanup
    if os.path.exists(PATH):
        os.remove(PATH)


def run_fake():
    import engram.backend as b
    import engram.consolidate as c
    import engram.engram as e
    import hashlib

    def fake_llm(prompt):
        if "new_episodes" in prompt:
            h = hashlib.md5(prompt.encode()).hexdigest()[:6]
            return ('{"new_episodes":[{"content":"memory %s","type":"fact",'
                    '"salience":0.7}],"reinforce":[],"supersede":[]}' % h)
        return "(fake reply: the unit is a trace)"

    def fake_embed(t):
        h = hashlib.md5(t.encode()).digest()
        return [x / 255.0 for x in h[:8]]

    b.llm = fake_llm; b.embed = fake_embed
    c.llm = fake_llm
    e.llm = fake_llm; e.embed = fake_embed
    print("Running offline with a deterministic fake backend.\n")
    run()


if __name__ == "__main__":
    if "--fake" in sys.argv:
        run_fake()
    else:
        run()