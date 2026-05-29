# Engram specialization experiment — run 2026-05-28 20:00

## What this run measured

Three claims, one experiment:

1. **Specialization** — does the chain make the LLM into a specific individual?
2. **Dynamics** — does the memory restructure itself as it accumulates, or just pile up?
3. **Persistence** — does the individual survive compression, and does the right thing change under sustained contradicting input?

## Headline numbers

- Specialization at full formation (after 30 messages of life): **mean distance 0.262** from the bare LLM.
  - identity facts: 0.327
  - dispositions:   0.200
  - synthesis:      0.255
- Memory dynamics at full formation: **29 new, 1 reinforce, 0 supersede, 0 gist** ops. Restructure ratio: 0.03.
- After the trauma phase (5 sustained anti-disposition messages):
  - identity facts: 0.310 (change from formation: -0.018)
  - dispositions:   0.200 (change: -0.000)
- At the end of compression (final checkpoint):
  - overall: 0.250 (change from formation: -0.012)
  - identity facts: 0.309
  - dispositions:   0.193
  - synthesis:      0.243

## How to read this

Specialization distance is the cosine distance between the loaded self's answer and what the *bare LLM* (no memory) says to the same question. 
Zero = the chain added nothing (no specialization). Higher = the chain has made the LLM into someone particular. The interesting story is whether and how that number moves across the run.

## Verifying this run

The full life of this self is recorded as a hash-linked chain in `engram_experiment_chain.json`. Anyone can verify it independently:

```bash
python -m engram.experiment --verify engram_experiment_chain.json
```

That command replays the chain from genesis, checks every block's hash, and confirms the chain has not been tampered with. The result of the experiment is bound to the chain — you can't fake the numbers without having actually lived the life.

Full per-probe results are in `engram_experiment_results.json`. Specialization chart in `engram_experiment_chart.png`.
