"""Monitor Agent: profile rules on per-unit signals; reports alarms active right now."""

from __future__ import annotations

import pandas as pd

from ..profiles import Profile


class MonitorAgent:
    def __init__(self, profile: Profile, debounce: int = 2):
        self.profile, self.debounce = profile, debounce

    def active(self, pu: pd.DataFrame) -> list[dict]:
        tail = pu.tail(self.debounce)
        out = []
        for code, a in self.profile.alarms.items():
            hit = a.crossed(tail[a.signal]) & self.profile.condition_mask(tail, a.condition)
            if len(tail) >= self.debounce and bool(hit.all()):
                out.append(
                    {"code": code, "name": a.name, "value_pu": float(tail[a.signal].iloc[-1]), "limit_pu": a.limit_pu}
                )
        return out
