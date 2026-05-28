"""
chain.py — the skeleton. The chain IS the self.

Each block commits to the previous one's hash. The unbroken line back to
genesis is what makes this one continuous self rather than a sequence of
unrelated states. This layer stays cryptographically exact and verifiable
even as the content layer (memory.py) forgets — lossy memory, lossless
identity.

A block stores a *delta*: the consolidated change an experience made to the
self (which episodes were born, reinforced, or superseded). Not the raw
experience — raw is referenced by hash and is droppable.
"""

import json
import time
import hashlib
from dataclasses import dataclass, field, asdict


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class Block:
    index: int
    prev_hash: str
    timestamp: float
    raw_ref: str          # hash of the full raw experience (stored off-chain, droppable)
    delta: list           # [{op, id, ...}] — what changed this block
    state_root: str       # hash over all live episodes after this block
    hash: str = ""

    def compute_hash(self) -> str:
        body = {k: v for k, v in asdict(self).items() if k != "hash"}
        return sha(json.dumps(body, sort_keys=True))


class Chain:
    def __init__(self):
        self.blocks: list[Block] = []

    def _state_root(self, memory) -> str:
        # deterministic fingerprint of the live self
        live = sorted(f"{e.id}:{e.salience:.4f}:{e.content}" for e in memory.live())
        return sha("|".join(live))

    def add_block(self, raw_experience: str, delta: list, memory) -> Block:
        prev = self.blocks[-1].hash if self.blocks else "GENESIS"
        b = Block(
            index=len(self.blocks),
            prev_hash=prev,
            timestamp=time.time(),
            raw_ref=sha(raw_experience),
            delta=delta,
            state_root=self._state_root(memory),
        )
        b.hash = b.compute_hash()
        self.blocks.append(b)
        return b

    def verify(self) -> str:
        """Re-derive every hash and check the links. Returns 'intact' or the fault."""
        for i, b in enumerate(self.blocks):
            if b.hash != b.compute_hash():
                return f"block {i} tampered (content changed after commit)"
            expected_prev = self.blocks[i - 1].hash if i > 0 else "GENESIS"
            if b.prev_hash != expected_prev:
                return f"block {i} broken link (prev_hash mismatch)"
        return "intact"

    def save(self, path: str):
        """Persist the chain — the entire source of truth — to a JSON file."""
        data = [asdict(b) for b in self.blocks]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Chain":
        """Load a chain from disk. Verify it before trusting the self it encodes."""
        c = cls()
        with open(path) as f:
            data = json.load(f)
        c.blocks = [Block(**b) for b in data]
        return c

    def __len__(self):
        return len(self.blocks)