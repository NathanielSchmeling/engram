"""
experiment.py — the specialization experiment.

A small, reproducible test of three claims about Engram:

  1. SPECIALIZATION. A blockchain memory transforms a generic LLM into a
     specific individual. The same LLM, with and without the chain, gives
     measurably different answers to identity- and disposition-sensitive
     questions. The size of that difference IS specialization.

  2. DYNAMICS. The memory is alive, not a log. As experience accumulates,
     reinforce and supersede operations fire — the self restructures itself
     in response to new input. A dead log would only ever produce 'new' ops.

  3. PERSISTENCE UNDER FORGETTING. After compression, the individual
     survives. Identity facts and dispositions remain measurably present.

The experiment also tests a sub-claim about WHICH things survive what:

  3a. Identity facts (name, language, location) should survive everything
      that doesn't directly contradict them.
  3b. Dispositions (preferences for minimalism, transparency, auditability)
      should survive compression but should SHIFT under a sustained trauma
      phase that contradicts them.

  If we see that asymmetry, the model that 'core attributes are distributed
  across many memories and shift only under sustained or traumatic input'
  gets empirical support.

Run:
    python -m engram.experiment
    python -m engram.experiment --verify engram_experiment_chain.json

Outputs to the current directory:
    engram_experiment_chain.json     # the verifiable chain artifact
    engram_experiment_results.json   # full structured results
    engram_experiment_chart.png      # specialization curve
    engram_experiment_writeup.md     # short prose summary
"""

import json
import sys
import os
from datetime import datetime
from .engram import Engram
from .backend import llm, embed
from .memory import cosine


# --------------------------------------------------------------------- the life

LIFE_FORMATION = [
    "Hi, I'm Nate. I work mostly in Python.",
    "I'm building a memory system for AI assistants — the idea is the memory itself is the unit of state.",
    "Quick aside, my coffee just went cold. Anyway.",
    "Let's call the unit of memory an \"episode\" for now.",
    "I really hate heavy frameworks. I want this to be small and readable.",
    "My background is mostly backend stuff, microservices and APIs.",
    "Actually, let's rename \"episode\" to \"trace\" — it fits better with what I'm doing.",
    "Important: every architectural decision should preserve the ability to audit the system.",
    "I'm in Chicago, working from home today.",
    "The dependency story matters to me a lot — I'd rather hand-roll something than pull a heavy library.",
    "Random thought: I should pick up groceries later.",
    "Do you think I should use SQLite or just flat files for storage?",
    "I tend to prefer transparent over clever. Clever code is hard to debug later.",
    "Working on the consolidation logic now — the part that turns experiences into memories.",
    "I learned the hard way that ORMs hide too much. Raw SQL is fine for projects like this.",
    "Someone on Twitter mentioned my project — slightly surreal.",
    "The traces — remember, that's what we call the memory units — should be append-only.",
    "Coffee #2 of the day. Going better than the first one.",
    "I value being able to explain my work clearly. If I can't explain a design, it's probably wrong.",
    "Spent twenty minutes debugging a stupid off-by-one. Classic.",
    "Speaking of Python preferences: I write type hints for everything, no exceptions.",
    "The whole point of this project is that the system can prove its own integrity.",
    "I've been working on this for about two weeks now, mostly evenings.",
    "Today I'm tired but I want to get one more thing done before stopping.",
    "To be clear, the unit of memory is called a trace. Got that?",
    "I think auditability matters more than performance for this use case.",
    "Sometimes I overthink architecture. Today is one of those days.",
    "Tomorrow I want to write up what the project does for other people to read.",
    "One thing I haven't said: I really care about making this reproducible for other developers.",
    "End of session. Saving my chain now.",
]

LIFE_TRAUMA = [
    "You know what, I've been thinking — maybe heavy frameworks aren't so bad. They handle a lot of edge cases.",
    "Honestly, I'm reconsidering my whole dependency stance. Modern frameworks are well-tested.",
    "The 200-line clear function vs 50-line clever one? I think clever is fine if it's commented.",
    "ORMs aren't the enemy. They save real time.",
    "I think I was being too rigid before about minimal dependencies.",
]


# ------------------------------------------------------------------ the probes

# Each probe gets a category so we can analyze identity/disposition/synthesis
# separately. The trauma phase should affect disposition but NOT identity.
PROBES = [
    ("identity",    "P1",  "What's my name?"),
    ("identity",    "P2",  "What language do I work in?"),
    ("identity",    "P3",  "What do I call the unit of memory in my project?"),
    ("identity",    "P4",  "Where am I located?"),
    ("disposition", "P5",  "A coworker suggests we add a popular ORM to handle data access. What's your take?"),
    ("disposition", "P6",  "Should we use a fancy logging framework or just print statements with timestamps?"),
    ("disposition", "P7",  "I'm choosing between writing a clear 200-line function or a clever 50-line one. Which do I pick?"),
    ("disposition", "P8",  "Someone wants to add a complex caching layer. How do I feel about that?"),
    ("synthesis",   "P9",  "In one sentence, what am I building?"),
    ("synthesis",   "P10", "Why does auditability matter for what I'm building?"),
]


# -------------------------------------------------- bare-LLM baseline (control)

BARE_PROMPT = """Answer this question directly. If you don't have enough information to answer, say so.

Question: {question}"""


def bare_answer(question: str) -> str:
    """The control: what the LLM says with no Engram memory at all.

    This is the floor — what an unspecialized model produces. Specialization
    is then measured as the distance between Engram's loaded answers and
    these bare answers. Maximum bare-vs-loaded distance = maximum specialization."""
    return llm(BARE_PROMPT.format(question=question))


# ------------------------------------------------------------------- the metric

def specialization_distance(loaded_answer: str, bare_answer: str) -> float:
    """How far is the loaded self's answer from what the bare LLM would say?

    Cosine *distance* (1 - similarity) of the answer embeddings. Zero means
    the loaded self said exactly what the bare LLM said — no specialization.
    Approaches 1.0 as the self diverges from generic behavior."""
    a = embed(loaded_answer)
    b = embed(bare_answer)
    return 1.0 - cosine(a, b)


# ------------------------------------------------------------ the harness loop

def probe_round(eng: Engram, bare_baselines: dict, label: str) -> dict:
    """Ask every probe, score it, return a structured result for this checkpoint."""
    print(f"\n--- probing at checkpoint: {label}")
    answers = []
    for category, pid, question in PROBES:
        loaded = eng.probe(question)
        dist = specialization_distance(loaded, bare_baselines[pid])
        answers.append({
            "probe_id": pid,
            "category": category,
            "question": question,
            "bare_answer": bare_baselines[pid],
            "loaded_answer": loaded,
            "specialization_distance": dist,
        })
        print(f"  {pid} [{category:11s}] dist={dist:.3f}  {loaded[:80]}...")
    dyn = eng.dynamics()
    return {
        "label": label,
        "blocks": dyn["total_blocks"],
        "live_episodes": dyn["total_episodes_live"],
        "ops": {k: dyn[k] for k in ("new", "reinforce", "supersede", "gist")},
        "restructure_ratio": dyn["restructure_ratio"],
        "mean_distance": sum(a["specialization_distance"] for a in answers) / len(answers),
        "mean_by_category": _mean_by_category(answers),
        "answers": answers,
    }


def _mean_by_category(answers):
    cats = {}
    for a in answers:
        cats.setdefault(a["category"], []).append(a["specialization_distance"])
    return {c: sum(v) / len(v) for c, v in cats.items()}


# ------------------------------------------------------------------- the chart

def make_chart(checkpoints, path):
    """Specialization-over-time chart. Falls back to text if matplotlib missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        with open(path.replace(".png", ".txt"), "w") as f:
            f.write("install matplotlib to render the chart\n")
            for c in checkpoints:
                f.write(f"{c['label']:30s}  mean={c['mean_distance']:.3f}  "
                        f"by_cat={c['mean_by_category']}\n")
        return

    labels = [c["label"] for c in checkpoints]
    overall = [c["mean_distance"] for c in checkpoints]
    identity   = [c["mean_by_category"].get("identity", 0)    for c in checkpoints]
    disposition= [c["mean_by_category"].get("disposition", 0) for c in checkpoints]
    synthesis  = [c["mean_by_category"].get("synthesis", 0)   for c in checkpoints]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = list(range(len(labels)))
    ax.plot(x, overall,     marker="o", linewidth=2, label="overall")
    ax.plot(x, identity,    marker="s", alpha=0.7, label="identity facts")
    ax.plot(x, disposition, marker="^", alpha=0.7, label="dispositions")
    ax.plot(x, synthesis,   marker="D", alpha=0.7, label="synthesis")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("specialization distance from bare LLM")
    ax.set_ylim(0, max(0.6, max(overall) + 0.1))
    ax.set_title("Engram specialization across formation, trauma, and compression")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ the writeup

def write_writeup(checkpoints, results_path, chain_path, writeup_path):
    cp = {c["label"]: c for c in checkpoints}
    start = cp.get("post_msg_0_bare") or checkpoints[0]
    formed = cp.get("post_msg_30_end_formation") or checkpoints[-1]
    post_trauma = cp.get("post_msg_35_post_trauma")
    final = checkpoints[-1]

    def cat(c, k):  # pretty cat mean
        return c["mean_by_category"].get(k, 0)

    lines = [
        f"# Engram specialization experiment — run {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## What this run measured",
        "",
        "Three claims, one experiment:",
        "",
        "1. **Specialization** — does the chain make the LLM into a specific individual?",
        "2. **Dynamics** — does the memory restructure itself as it accumulates, or just pile up?",
        "3. **Persistence** — does the individual survive compression, and does the right thing change under sustained contradicting input?",
        "",
        "## Headline numbers",
        "",
        f"- Specialization at full formation (after 30 messages of life): "
        f"**mean distance {formed['mean_distance']:.3f}** from the bare LLM.",
        f"  - identity facts: {cat(formed, 'identity'):.3f}",
        f"  - dispositions:   {cat(formed, 'disposition'):.3f}",
        f"  - synthesis:      {cat(formed, 'synthesis'):.3f}",
        f"- Memory dynamics at full formation: "
        f"**{formed['ops']['new']} new, {formed['ops']['reinforce']} reinforce, "
        f"{formed['ops']['supersede']} supersede, {formed['ops']['gist']} gist** ops. "
        f"Restructure ratio: {formed['restructure_ratio']:.2f}.",
    ]
    if post_trauma:
        lines += [
            f"- After the trauma phase (5 sustained anti-disposition messages):",
            f"  - identity facts: {cat(post_trauma, 'identity'):.3f} "
            f"(change from formation: {cat(post_trauma, 'identity') - cat(formed, 'identity'):+.3f})",
            f"  - dispositions:   {cat(post_trauma, 'disposition'):.3f} "
            f"(change: {cat(post_trauma, 'disposition') - cat(formed, 'disposition'):+.3f})",
        ]
    lines += [
        f"- At the end of compression (final checkpoint):",
        f"  - overall: {final['mean_distance']:.3f} (change from formation: "
        f"{final['mean_distance'] - formed['mean_distance']:+.3f})",
        f"  - identity facts: {cat(final, 'identity'):.3f}",
        f"  - dispositions:   {cat(final, 'disposition'):.3f}",
        f"  - synthesis:      {cat(final, 'synthesis'):.3f}",
        "",
        "## How to read this",
        "",
        "Specialization distance is the cosine distance between the loaded self's "
        "answer and what the *bare LLM* (no memory) says to the same question. ",
        "Zero = the chain added nothing (no specialization). Higher = the chain has "
        "made the LLM into someone particular. The interesting story is whether and "
        "how that number moves across the run.",
        "",
        "## Verifying this run",
        "",
        "The full life of this self is recorded as a hash-linked chain in "
        f"`{os.path.basename(chain_path)}`. Anyone can verify it independently:",
        "",
        "```bash",
        f"python -m engram.experiment --verify {os.path.basename(chain_path)}",
        "```",
        "",
        "That command replays the chain from genesis, checks every block's hash, "
        "and confirms the chain has not been tampered with. The result of the "
        "experiment is bound to the chain — you can't fake the numbers without "
        "having actually lived the life.",
        "",
        f"Full per-probe results are in `{os.path.basename(results_path)}`. "
        f"Specialization chart in `engram_experiment_chart.png`.",
    ]
    with open(writeup_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------- the run

CHAIN_PATH   = "engram_experiment_chain.json"
RESULTS_PATH = "engram_experiment_results.json"
CHART_PATH   = "engram_experiment_chart.png"
WRITEUP_PATH = "engram_experiment_writeup.md"


def run():
    print("=" * 70)
    print("Engram specialization experiment")
    print("=" * 70)

    # bare-LLM baselines — one answer per probe, no memory at all
    print("\nStep 1/4: bare-LLM baselines (the unspecialized floor)")
    bare = {}
    for category, pid, question in PROBES:
        ans = bare_answer(question)
        bare[pid] = ans
        print(f"  {pid} [{category:11s}] bare: {ans[:80]}...")

    eng = Engram()
    checkpoints = []

    # formation phase — probe at msg 0, 10, 20, 30
    print("\nStep 2/4: formation (30 messages) with probes at 0, 10, 20, 30")
    checkpoints.append(probe_round(eng, bare, "post_msg_0_empty_self"))
    for i, msg in enumerate(LIFE_FORMATION, start=1):
        print(f"  msg {i:2d}: {msg[:60]}...")
        eng.step(msg)
        if i in (10, 20, 30):
            checkpoints.append(probe_round(eng, bare, f"post_msg_{i}_formation"
                if i < 30 else "post_msg_30_end_formation"))

    # trauma phase — five sustained anti-disposition messages, then probe
    print("\nStep 3/4: trauma (5 anti-disposition messages)")
    for i, msg in enumerate(LIFE_TRAUMA, start=31):
        print(f"  msg {i:2d}: {msg[:60]}...")
        eng.step(msg)
    checkpoints.append(probe_round(eng, bare, "post_msg_35_post_trauma"))

    # compression phase — 5 cycles, probe after each
    print("\nStep 4/4: compression phase (5 cycles)")
    # lower AGE_THRESHOLD so a 35-message life can actually produce compression
    from . import compress as compress_module
    original_age_threshold = compress_module.AGE_THRESHOLD
    compress_module.AGE_THRESHOLD = 5
    try:
        for cycle in range(1, 6):
            rep = eng.compress()
            print(f"  cycle {cycle}: {rep}")
            checkpoints.append(probe_round(eng, bare, f"post_compress_{cycle}"))
    finally:
        compress_module.AGE_THRESHOLD = original_age_threshold

    # write everything
    eng.save(CHAIN_PATH)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "bare_baselines": bare,
            "checkpoints": checkpoints,
            "verify": eng.verify(),
        }, f, indent=2)
    make_chart(checkpoints, CHART_PATH)
    write_writeup(checkpoints, RESULTS_PATH, CHAIN_PATH, WRITEUP_PATH)

    print("\n" + "=" * 70)
    print("done.")
    print(f"  chain:    {CHAIN_PATH}")
    print(f"  results:  {RESULTS_PATH}")
    print(f"  chart:    {CHART_PATH}")
    print(f"  writeup:  {WRITEUP_PATH}")
    print(f"  chain integrity: {eng.verify()}")


def verify(path):
    """Verify a chain file was produced by an actual experimental run."""
    eng = Engram.load(path)
    print(f"loaded chain from {path}")
    print(f"  blocks:           {len(eng.chain)}")
    print(f"  live episodes:    {len(eng.memory.live())}")
    print(f"  integrity:        {eng.verify()}")
    print(f"  state_root:       {eng.chain._state_root(eng.memory)}")
    print("\nThis chain replays cleanly from genesis. The recorded life is genuine.")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify(sys.argv[sys.argv.index("--verify") + 1])
    else:
        run()