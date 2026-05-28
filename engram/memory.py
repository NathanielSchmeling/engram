"""
memory.py — episodes and retrieval.

An Episode is one consolidated memory: the change an experience left behind.
The MemoryStore holds all live episodes and serves salience+recency-weighted
retrieval. This is the content layer — it forgets like a mind. The chain
(chain.py) is the skeleton that stays exact.

Append-only discipline: episodes are never edited in place. They are born,
reinforced (salience bumped), or superseded (tombstoned, not deleted). Dead
episodes survive as hashes so the chain stays verifiable.
"""

from dataclasses import dataclass, field, asdict


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Episode:
    id: str
    content: str
    type: str                 # fact | preference | event | insight
    salience: float           # how strongly it registered; drives reinforcement & decay
    embedding: list[float]
    birth_block: int
    last_seen_block: int
    dead: bool = False        # tombstoned by supersede; kept for verifiability
    gist: bool = False        # produced by the compression pass

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding")    # keep summaries readable
        return d


class MemoryStore:
    # retrieval weighting: meaning matters most, then importance, then recency
    W_SIM, W_SAL, W_REC = 0.60, 0.25, 0.15

    def __init__(self):
        self.episodes: dict[str, Episode] = {}
        self._counter = 0

    def _new_id(self) -> str:
        self._counter += 1
        return f"ep_{self._counter}"

    def live(self) -> list[Episode]:
        return [e for e in self.episodes.values() if not e.dead]

    # --- the three append-only operations -----------------------------------

    def add(self, content, etype, salience, embedding, block_index) -> Episode:
        e = Episode(
            id=self._new_id(), content=content, type=etype,
            salience=max(0.0, min(1.0, salience)), embedding=embedding,
            birth_block=block_index, last_seen_block=block_index,
        )
        self.episodes[e.id] = e
        return e

    def reinforce(self, eid, delta, block_index) -> bool:
        e = self.episodes.get(eid)
        if not e or e.dead:
            return False
        e.salience = max(0.0, min(1.0, e.salience + delta))
        e.last_seen_block = block_index
        return True

    def supersede(self, eid) -> bool:
        e = self.episodes.get(eid)
        if not e or e.dead:
            return False
        e.dead = True
        return True

    # --- retrieval ----------------------------------------------------------

    # any memory in the TOP QUARTILE of similarity for the current query is
    # guaranteed to surface, regardless of salience/recency. relative rather
    # than absolute because absolute cosine scales differ across embedding
    # models — when one model writes the memories and another queries them
    # (as in the LLM-swap case), the same semantically-correct match may
    # come back at 0.3 instead of 0.7 even though the *ranking* is the same.
    # the right thing to guarantee is "if it's near the top of the field,
    # don't let salience/recency push it out."
    SIM_TOP_FRACTION = 0.25

    def retrieve(self, query_embedding, current_block, k=8) -> list[Episode]:
        scored = []
        for e in self.live():
            sim = cosine(query_embedding, e.embedding)
            recency = 1.0 / (1 + current_block - e.last_seen_block)
            score = self.W_SIM * sim + self.W_SAL * e.salience + self.W_REC * recency
            scored.append((score, sim, e))
        if not scored:
            return []
        scored.sort(key=lambda x: x[0], reverse=True)

        # relative floor: top fraction by similarity is guaranteed to surface
        sims_desc = sorted((s for _, s, _ in scored), reverse=True)
        cutoff_idx = max(0, int(len(sims_desc) * self.SIM_TOP_FRACTION) - 1)
        sim_cutoff = sims_desc[cutoff_idx] if sims_desc else 1.0
        guaranteed = {e.id for _, sim, e in scored if sim >= sim_cutoff}

        chosen, used = [], set()
        for _, _, e in scored:
            if e.id in guaranteed and e.id not in used:
                chosen.append(e); used.add(e.id)
        for _, _, e in scored:
            if len(chosen) >= k:
                break
            if e.id not in used:
                chosen.append(e); used.add(e.id)
        return chosen

    def most_similar(self, query_embedding, threshold=0.92) -> Episode | None:
        """Find a live episode whose embedding is above `threshold` similar to
        the query, or None. Used by consolidation to deduplicate: a near-
        identical new episode should reinforce the existing one, not create
        a near-duplicate."""
        best, best_sim = None, threshold
        for e in self.live():
            s = cosine(query_embedding, e.embedding)
            if s > best_sim:
                best, best_sim = e, s
        return best