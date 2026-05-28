"""
engram.py — the system.

Ties everything together: one user message in, consolidate -> commit to chain
-> recall -> respond. The Engram object IS the continuous self; the chain
inside it is the unbroken thread that makes it one.
"""

from .backend import llm, embed
from .memory import MemoryStore
from .chain import Chain
from .consolidate import consolidate
from .compress import compress
from .replay import rebuild_from_chain

REPLY_PROMPT = """You are an AI with persistent memory. Below are the memories
you have that are relevant to the current message. Older memories may survive
only as gist summaries.

YOUR MEMORIES:
{memories}

User: {message}

Respond grounded ONLY in the memories above. If a memory contains the answer,
state it directly. If your memories don't contain enough to answer a factual
question, say so plainly rather than guessing. Do not invent past projects,
prior conversations, or details that aren't in the memories above."""


class Engram:
    def __init__(self):
        self.memory = MemoryStore()
        self.chain = Chain()

    def step(self, message: str) -> str:
        """One experience = one block. The core loop."""
        # 1. learn: turn the experience into a consolidated delta
        delta = consolidate(message, self.memory, self.chain, embed)
        # 2. commit: advance the chain (even an empty delta advances continuity)
        self.chain.add_block(message, delta, self.memory)
        # 3. recall: retrieve the self's relevant memories (k=8: more context
        #    is cheap; missing the answer memory is expensive)
        recalled = self.memory.retrieve(embed(message), len(self.chain), k=8)
        mem_str = "\n".join(f"- {e.content}" for e in recalled) or "- (no memories yet)"
        # 4. respond: act as a self shaped by what it registered
        return llm(REPLY_PROMPT.format(memories=mem_str, message=message))

    def compress(self) -> dict:
        return compress(self.memory, self.chain, embed)

    def rebuild_cache(self):
        """Throw away the episode store and reconstruct it from the chain alone.

        This is the proof that the chain is the memory: after this call the self
        is identical, but every episode was rebuilt purely by replaying blocks.
        """
        self.memory = rebuild_from_chain(self.chain, embed)
        return self.memory

    def save(self, path: str = "engram_chain.json"):
        """Persist the self. Because the chain is the source of truth, saving
        the chain saves the entire self — the episode store is disposable."""
        self.chain.save(path)
        return path

    @classmethod
    def load(cls, path: str = "engram_chain.json") -> "Engram":
        """Wake a self from disk: load the chain, verify it, replay it.

        The self that boots is provably the same one that shut down — the chain
        is verified intact before its self is reconstructed, so a tampered file
        is caught before it can poison the self.
        """
        eng = cls()
        eng.chain = Chain.load(path)
        integrity = eng.chain.verify()
        if integrity != "intact":
            raise ValueError(f"refusing to wake a corrupted self: {integrity}")
        eng.rebuild_cache()
        return eng

    def verify(self) -> str:
        return self.chain.verify()

    def snapshot(self) -> dict:
        """Human-readable view of the current self and its skeleton."""
        return {
            "blocks": len(self.chain),
            "integrity": self.verify(),
            "live_episodes": [e.to_dict() for e in self.memory.live()],
        }