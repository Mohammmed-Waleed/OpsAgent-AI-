"""Fault injection for demonstrating and testing crash recovery.

``SimulatedCrash`` derives from BaseException so nothing in the agent swallows it, and the
tracer deliberately leaves the current span open - exactly what a SIGKILL would leave behind.
Spec format: ``<point>[:<tool>]`` where point is one of
after_llm | after_approval | mid_tool | after_tools.  Fires once.
"""

from __future__ import annotations

POINTS = ("after_llm", "after_approval", "mid_tool", "after_tools")


class SimulatedCrash(BaseException):
    pass


class Faults:
    def __init__(self, spec: str | None = None):
        self.point, _, self.tool = (spec or "").partition(":")
        if self.point and self.point not in POINTS:
            raise ValueError(f"unknown crash point {self.point!r}; choose from {', '.join(POINTS)}")
        self.fired = False

    def hit(self, point: str, tool: str | None = None) -> None:
        if self.fired or point != self.point or (self.tool and self.tool != tool):
            return
        self.fired = True
        raise SimulatedCrash(f"simulated crash at {point}" + (f" ({tool})" if tool else ""))
