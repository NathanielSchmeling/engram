"""
consolidate.py — the heart.

Given the current relevant memories and a new experience, decide what changed:
what new episodes formed, which existing ones got reinforced, which are now
outdated. This is where 'consensus = consolidation' lives, collapsed into a
single deterministic-enough function.

Three operations only — new, reinforce, supersede — which keeps the schema
clean and the chain append-only. Nothing is edited in place.
"""

from .backend import llm, parse_json

PROMPT = """You extract memories from a conversation with a user.

What you currently remember that's relevant:
{memories}

The user just said:
"{message}"

Output ONLY a JSON object. Here is a worked example showing the format (the
example users 'PERSON_X' and 'PERSON_Y' are placeholders — never copy them
into a real memory):

{{"new_episodes": [{{"content": "PERSON_X's name is PERSON_X", "type": "fact", "salience": 0.9}}, {{"content": "PERSON_X dislikes meetings before noon", "type": "preference", "salience": 0.7}}], "reinforce": [{{"id": "ep_3", "salience_delta": 0.2}}], "supersede": [{{"id": "ep_5", "reason": "user moved cities"}}]}}

Now do the same for the user's actual message above. Rules:
- Write real memories about THIS user, based ONLY on what they actually said.
- If the user states their name, role, location, or other identity fact, ALWAYS create a separate memory for it with salience 0.9 or higher. Identity facts must never be folded into other memories.
- NEVER use the placeholder names PERSON_X or PERSON_Y in your output. If you don't know the user's name, just say "the user".
- content: a specific fact about the user or project, in your own words.
- type: one of fact, preference, event, insight.
- salience: 0.0 to 1.0, how important it is to remember.
- Before creating a new episode, check the memories listed above. If the message just restates something you already remember, use reinforce with that memory's id instead of creating a near-duplicate.
- If the user renames or changes something (like "call them X instead of Y"), you MUST supersede the old memory's id AND create the new one. A rename is always supersede + new.
- If the message contradicts something you remember, put that memory's id in supersede.
- If nothing worth remembering, use empty arrays: {{"new_episodes": [], "reinforce": [], "supersede": []}}
- Output the JSON and nothing else."""


def consolidate(message: str, memory, chain, embed) -> list:
    """Run consolidation and apply it to `memory`. Returns the delta for the block."""
    block_index = len(chain)
    relevant = memory.retrieve(embed(message), block_index, k=8)
    mem_str = "\n".join(f"{e.id}: {e.content}" for e in relevant) or "(none yet)"

    out = parse_json(llm(PROMPT.format(memories=mem_str, message=message)))
    delta = []

    # reject template echoes that occasionally slip through on small models
    JUNK = {"concise memory", "the merged gist",
            "person_x's name is person_x",
            "person_x dislikes meetings before noon",
            "user's name is alex",
            "user dislikes meetings before noon",
            ""}

    for ne in out["new_episodes"]:
        content = (ne.get("content") or "").strip()
        if content.lower() in JUNK:
            continue
        etype = ne.get("type", "event")
        if "|" in etype:                      # model echoed the type template
            etype = "event"

        # deterministic dedup: if a near-identical live memory already exists,
        # reinforce it instead of adding a near-duplicate. This catches the
        # case the prompt rule misses — same fact phrased slightly differently
        # across turns, where the model treats it as new.
        candidate_emb = embed(content)
        twin = memory.most_similar(candidate_emb, threshold=0.92)
        if twin is not None:
            sdelta = 0.1
            if memory.reinforce(twin.id, sdelta, block_index):
                delta.append({"op": "reinforce", "id": twin.id,
                              "salience_delta": sdelta})
            continue

        e = memory.add(
            content=content,
            etype=etype,
            salience=float(ne.get("salience", 0.5)),
            embedding=candidate_emb,
            block_index=block_index,
        )
        delta.append({"op": "new", "id": e.id, "content": e.content,
                      "type": e.type, "salience": e.salience})

    for r in out["reinforce"]:
        sdelta = float(r.get("salience_delta", 0.1))
        if memory.reinforce(r.get("id"), sdelta, block_index):
            delta.append({"op": "reinforce", "id": r["id"], "salience_delta": sdelta})

    for s in out["supersede"]:
        if memory.supersede(s.get("id")):
            delta.append({"op": "supersede", "id": s["id"],
                          "reason": s.get("reason", "")})

    return delta