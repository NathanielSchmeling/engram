"""
replay.py — the chain IS the memory.

This is the module that earns the thesis. The episode store is no longer the
source of truth; it is a *cache* that can be reconstructed entirely by replaying
the chain from genesis. Delete the store, replay the blocks, and the identical
live self comes back — because the self was never in the store. It was in the
chain the whole time.

The rule that keeps this true: nothing changes the self except by appending a
block. Every birth, reinforcement, supersession (and later, every act of
forgetting) is a block. If a change happens outside the chain, replay won't
reproduce it and the chain stops being the memory.
"""

from .memory import MemoryStore, Episode


def rebuild_from_chain(chain, embed) -> MemoryStore:
    """Reconstruct the live self by replaying every block's delta in order.

    Uses the ids recorded in the chain directly, so reinforce/supersede ops
    reference the same episodes they did originally. The store is pure output;
    the chain is the only input.
    """
    mem = MemoryStore()

    for block in chain.blocks:
        for op in block.delta:
            kind = op.get("op")

            if kind == "new":
                eid = op["id"]
                e = Episode(
                    id=eid,
                    content=op["content"],
                    type=op.get("type", "event"),
                    salience=float(op.get("salience", 0.5)),
                    embedding=embed(op["content"]),
                    birth_block=block.index,
                    last_seen_block=block.index,
                )
                mem.episodes[eid] = e
                # keep the counter ahead of any replayed id so future live
                # writes don't collide with reconstructed ones
                n = _id_number(eid)
                if n is not None:
                    mem._counter = max(mem._counter, n)

            elif kind == "reinforce":
                mem.reinforce(op["id"], float(op.get("salience_delta", 0.1)),
                              block.index)

            elif kind == "supersede":
                mem.supersede(op["id"])

            elif kind == "gist":
                # a compression event: born-as-gist episode that merges others
                eid = op["id"]
                e = Episode(
                    id=eid,
                    content=op["content"],
                    type="insight",
                    salience=float(op.get("salience", 0.4)),
                    embedding=embed(op["content"]),
                    birth_block=block.index,
                    last_seen_block=block.index,
                    gist=True,
                )
                mem.episodes[eid] = e
                for merged_id in op.get("merged", []):
                    mem.supersede(merged_id)
                n = _id_number(eid)
                if n is not None:
                    mem._counter = max(mem._counter, n)

    return mem


def _id_number(eid: str):
    """Extract the integer from an id like 'ep_7'. Returns None if unparseable."""
    try:
        return int(eid.split("_")[1])
    except (IndexError, ValueError):
        return None