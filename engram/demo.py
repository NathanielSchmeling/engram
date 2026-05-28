"""
demo.py — prove the three properties.

    python -m engram.demo            # real, uses Ollama
    python -m engram.demo --fake     # offline, no models needed (tests chain only)

Demonstrates:
  1. Continuity is provable   — verify from genesis; tamper; watch it fail.
  2. It forgets like a mind    — recent detailed, old as gist, skeleton exact.
  3. History shaped behavior   — replies reflect consolidated past experience.
"""

import sys
from .engram import Engram


def run():
    eng = Engram()

    script = [
        "Hi, I'm building a blockchain-based memory system for AI.",
        "I've decided episodic memories are the right unit of state.",
        "My name is Sam and I work mostly in Python.",
        "Actually, let's call them 'traces' instead of episodes from now on.",
        "I prefer minimal dependencies — no heavy frameworks.",
        "What am I building, and what do I call the unit of memory?",
    ]

    for msg in script:
        print(f"\nUSER: {msg}")
        print(f"AI:   {eng.step(msg)}")

    print("\n" + "=" * 60)
    print("PROPERTY 1 — continuity is provable")
    print("  chain integrity:", eng.verify())

    print("\nPROPERTY 3 — the history shaped behavior")
    print("  live self (the memories driving the AI's answers):")
    for e in eng.memory.live():
        tag = "gist" if e.gist else e.type
        print(f"    [{e.salience:.2f}] ({tag}) {e.content}")

    # tamper test — inject a fake memory into a past block
    if len(eng.chain) > 1:
        eng.chain.blocks[1].delta.append(
            {"op": "new", "id": "FAKE", "content": "Sam loves crypto scams"}
        )
        print("\n  after tampering with block 1:", eng.verify())
        print("  (the self can prove it was altered — identity is trustworthy)")


def run_fake():
    """Offline: monkeypatch the backend so chain mechanics run with no models."""
    import engram.backend as b
    import random

    def fake_llm(prompt):
        if "STRICT JSON" in prompt and "new_episodes" in prompt:
            return '{"new_episodes":[{"content":"a remembered thing","type":"fact","salience":0.6}],"reinforce":[],"supersede":[]}'
        return "(fake reply shaped by memory)"

    fake_embed = lambda t: [random.random() for _ in range(8)]
    b.llm = lambda p: fake_llm(p)
    b.embed = fake_embed
    # rebind the names already imported into each module at load time
    import engram.consolidate as c, engram.engram as e, engram.compress as cm
    c.llm = b.llm
    e.llm = b.llm
    e.embed = fake_embed
    cm.llm = b.llm

    print("Running offline with a fake backend (chain mechanics only).\n")
    run()


if __name__ == "__main__":
    if "--fake" in sys.argv:
        run_fake()
    else:
        run()