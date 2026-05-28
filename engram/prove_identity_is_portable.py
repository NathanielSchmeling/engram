"""
prove_identity_is_portable.py — the capstone. Answers: do we need our own LLM?

Build a self using one model. Persist it (the chain only). Then wake it using a
DIFFERENT model and check that the same self comes back.

If it does, identity was never in the model. The LLM supplies the generic
faculty of cognition — the how-of-thinking — and is swappable. The chain
supplies the who — the specific self — and is what persists. A brain can't do
this: its memories ARE its processor. Engram separates them, so the self is
portable across cognition substrates.

What is and isn't preserved across a swap (and why that's correct):
  - PRESERVED: the episodic self — every memory, its type, its salience, the
    tombstones, the gist structure. This is hashed into state_root, so a match
    is cryptographic proof the self is identical.
  - REGENERATED: the embedding vectors, recomputed by the new model on replay.
    The self's *content* is model-independent; only its *retrieval geometry*
    (how it searches memories) is model-dependent. Identity portable, cognition
    swappable — exactly the right division.

Real run (requires two models pulled in Ollama):
    ollama pull llama3.2:3b
    ollama pull qwen2.5:7b
    python -m engram.prove_identity_is_portable --build llama3.2:3b
    python -m engram.prove_identity_is_portable --wake  qwen2.5:7b

Offline structural proof (no models; simulates two different backends):
    python -m engram.prove_identity_is_portable --fake
"""

import sys
from .engram import Engram

PATH = "engram_portable_self.json"


def fingerprint(memory):
    """Content + salience + lifecycle — the self, independent of embeddings."""
    return sorted(f"{e.content}|{e.type}|{e.salience:.4f}|{e.dead}|{e.gist}"
                  for e in memory.episodes.values())


def build_self_on(model_name):
    """Session 1: build and persist a self using `model_name`."""
    import engram.backend as b
    b.LLM_MODEL = model_name
    eng = Engram()
    for m in [
        "Hi, I'm building a blockchain memory system for AI.",
        "My name is Nate and I work mostly in Python.",
        "Let's call the unit of memory a 'trace'.",
        "I prefer minimal dependencies.",
    ]:
        eng.step(m)
    eng.save(PATH)
    root = eng.chain._state_root(eng.memory)
    print(f"[build on {model_name}] {len(eng.memory.live())} memories, "
          f"{len(eng.chain)} blocks")
    print(f"[build on {model_name}] state_root: {root}")
    print(f"[build on {model_name}] saved to {PATH}")
    return fingerprint(eng.memory), root


def wake_self_on(model_name):
    """Session 2: wake the persisted self using a DIFFERENT model."""
    import engram.backend as b
    b.LLM_MODEL = model_name
    eng = Engram.load(PATH)            # verifies chain, replays to rebuild self
    root = eng.chain._state_root(eng.memory)
    # capture the self EXACTLY as it woke, before it lives any further
    woke_fp = fingerprint(eng.memory)
    print(f"[wake on {model_name}] {len(eng.memory.live())} memories, "
          f"{len(eng.chain)} blocks")
    print(f"[wake on {model_name}] state_root: {root}")
    print(f"[wake on {model_name}] integrity: {eng.verify()}")
    # now let the woken self answer from memory, thinking with a different model
    reply = eng.step("What do I call the unit of memory, and what's my name?")
    print(f"[wake on {model_name}] the revived self says:\n  {reply[:240]}")
    return woke_fp, root, eng


def verdict(build_fp, build_root, wake_fp, wake_root):
    print("\n" + "=" * 60)
    same = (build_fp == wake_fp) and (build_root == wake_root)
    if same:
        print("VERDICT: SAME SELF across a model swap.")
        print("Identity lived in the chain, not the model. The LLM was renting")
        print("cognition to a self that persists independently of it.")
        print("=> You do NOT need to build your own LLM.")
    else:
        print("VERDICT: the self changed across the swap — identity is")
        print("entangled with the model somewhere it shouldn't be.")
        lost = set(build_fp) - set(wake_fp)
        gained = set(wake_fp) - set(build_fp)
        if lost:   print("  lost:  ", list(lost)[:3])
        if gained: print("  gained:", list(gained)[:3])


def run_fake():
    """Structural proof with two DIFFERENT deterministic fake backends.

    The 'build' backend and 'wake' backend produce different embeddings (like
    two real models would) but the same episodic content. We show the self
    survives the swap because the self is content, not embeddings.
    """
    import engram.backend as b, engram.consolidate as c
    import engram.engram as e, engram.compress as cm
    import hashlib, os

    n = [0]
    def fake_llm(p):
        if "new_episodes" in p:
            n[0] += 1
            facts = ["User's name is Nate", "User works in Python",
                     "The unit of memory is called a trace",
                     "User prefers minimal dependencies"]
            f = facts[(n[0] - 1) % len(facts)]
            return ('{"new_episodes":[{"content":"%s","type":"fact",'
                    '"salience":0.8}],"reinforce":[],"supersede":[]}' % f)
        return "Your name is Nate and you call the unit of memory a trace."

    # two DIFFERENT embedding functions — simulate two different models
    def embed_model_A(t):
        return [x/255.0 for x in hashlib.md5(("A"+t).encode()).digest()[:8]]
    def embed_model_B(t):
        return [x/255.0 for x in hashlib.sha1(("B"+t).encode()).digest()[:8]]

    print("Running offline. Two different fake backends stand in for two models.")
    print("They produce different EMBEDDINGS but the same MEMORIES.\n")

    # session 1: build on "model A"
    b.llm = fake_llm; c.llm = fake_llm; e.llm = fake_llm; cm.llm = fake_llm
    b.embed = embed_model_A; e.embed = embed_model_A
    build_fp, build_root = build_self_on("fake-model-A")

    # session 2: wake on "model B" — different embeddings entirely
    b.embed = embed_model_B; e.embed = embed_model_B
    wake_fp, wake_root, _ = wake_self_on("fake-model-B")

    verdict(build_fp, build_root, wake_fp, wake_root)
    if os.path.exists(PATH):
        os.remove(PATH)


if __name__ == "__main__":
    if "--fake" in sys.argv:
        run_fake()
    elif "--build" in sys.argv:
        model = sys.argv[sys.argv.index("--build") + 1]
        build_self_on(model)
    elif "--wake" in sys.argv:
        model = sys.argv[sys.argv.index("--wake") + 1]
        # to compare, the user runs --build first; we just report the woken self
        wake_fp, wake_root, _ = wake_self_on(model)
        print("\n(Compare this state_root to the one printed during --build.)")
    else:
        print(__doc__)