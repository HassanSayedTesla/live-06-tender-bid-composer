"""Memory inspection + persistence. Ch3.2 / Ch3.3 pattern."""
from __future__ import annotations

import json
from pathlib import Path


def save_agent_memory(agent, path: str | Path = "agent_memory.json") -> Path:
    path = Path(path)
    try:
        steps = agent.memory.get_succinct_steps()
    except Exception:
        steps = []
    path.write_text(json.dumps(steps, indent=2, default=str), encoding="utf-8")
    return path


def full_code(agent) -> str:
    try:
        return agent.memory.return_full_code()
    except Exception as e:
        return f"<no full code: {e}>"
