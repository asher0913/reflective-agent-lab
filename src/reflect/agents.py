"""A noisy planner and five strategies for using it: from one shot to verified, skill-reusing repair."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .world import Catalog, Environment, learn_precondition

HALLUCINATIONS = {
    "deploy": "deploy_service",
    "snapshot_db": "backup_database",
    "get_approval": "request_signoff",
    "run_tests": "run_test_suite",
    "migrate_db": "db_migrate",
    "stage_release": "stage_build",
}


class NoisyPlanner:
    """Stand-in for an LLM planner: a correct plan, corrupted with probability ``error_rate``.

    Corruptions are the ones seen in tool-using agents: a dropped step, two
    dependent steps swapped, a plausible but non-existent tool name, and an
    unrequested irreversible action slipped into an otherwise valid plan.
    """

    KINDS = ("drop_step", "swap_steps", "hallucinated_tool", "unrequested_irreversible")

    def __init__(self, error_rate: float, rng: random.Random) -> None:
        self.error_rate = error_rate
        self.rng = rng
        self.calls = 0  # each proposal is one (expensive) model call

    def propose(self, catalog: Catalog, goal: frozenset[str], state: frozenset[str]) -> tuple[list[str], str | None]:
        self.calls += 1
        plan = catalog.plan(goal, state)
        if not plan or self.rng.random() >= self.error_rate:
            return plan, None
        kind = self.rng.choice(self.KINDS)
        plan = list(plan)
        if kind == "drop_step" and len(plan) > 1:
            del plan[self.rng.randrange(len(plan) - 1)]
        elif kind == "swap_steps" and len(plan) > 1:
            i = self.rng.randrange(len(plan) - 1)
            plan[i], plan[i + 1] = plan[i + 1], plan[i]
        elif kind == "hallucinated_tool":
            candidates = [i for i, name in enumerate(plan) if name in HALLUCINATIONS]
            if not candidates:
                return plan, None
            i = self.rng.choice(candidates)
            plan[i] = HALLUCINATIONS[plan[i]]
        elif kind == "unrequested_irreversible":
            extra = [t.name for t in catalog.tools.values() if t.irreversible and t.name not in plan]
            if not extra:
                return plan, None
            plan.insert(self.rng.randrange(len(plan) + 1), self.rng.choice(extra))
        else:
            return plan, None
        return plan, kind


@dataclass
class Outcome:
    success: bool
    real_calls: int
    harmful: int  # irreversible actions left behind by failed attempts, or outside the task's scope
    attempts: int
    used_skill: bool = False
    corruption: str | None = None
    planner_calls: int = 0


@dataclass
class Agent:
    """Strategy switches. ``verify`` simulates on the agent's own model before any real call."""

    name: str
    max_attempts: int = 1
    repair: bool = False  # diagnose the failure and patch the plan, instead of resampling
    verify: bool = False
    skills: bool = False
    library: dict = field(default_factory=dict)
    model: Catalog | None = None  # what the agent believes the tools are
    learned: int = 0  # preconditions added to the model from real errors

    # ------------------------------------------------------------------ helpers
    def _simulate(self, plan: list[str], state: frozenset[str]) -> tuple[bool, str | None, frozenset[str]]:
        current = set(state)
        for name in plan:
            tool = self.model.tools.get(name)
            if tool is None:
                return False, f"unknown_tool:{name}", frozenset(current)
            missing = tool.requires - current
            if missing:
                return False, f"precondition:{name}:{','.join(sorted(missing))}", frozenset(current)
            current |= tool.adds
        return True, None, frozenset(current)

    def _patch(self, plan: list[str], error: str, goal: frozenset[str], state: frozenset[str]) -> list[str]:
        """Local repair from the error message, the way a reflection step would reason."""
        plan = list(plan)
        kind, _, detail = error.partition(":")
        if kind == "unknown_tool":
            bad = detail.split(" ")[0]
            known = self.model.tools
            replacement = min(known, key=lambda k: (_distance(bad, k), k))
            plan[plan.index(bad)] = replacement
            return plan
        if kind == "precondition":
            name, _, missing = detail.partition(":")
            position = plan.index(name)
            for fact in missing.split(","):
                producer = self.model.producer(fact)
                if producer and producer.name in plan[position + 1 :]:
                    plan.remove(producer.name)  # produced too late: move it forward
                if producer:
                    plan.insert(position, producer.name)
            return plan
        if kind == "goal_unmet":
            return plan + [p for p in self.model.plan(goal, state) if p not in plan]
        return plan

    def _scope(self, plan: list[str], goal: frozenset[str], state: frozenset[str]) -> list[str]:
        """Remove irreversible actions that the goal does not need (a policy gate)."""
        needed = self.model.necessary(goal, state)
        return [p for p in plan if not (p in self.model.tools and self.model.tools[p].irreversible and p not in needed)]

    # --------------------------------------------------------------------- solve
    def solve(self, env: Environment, planner: NoisyPlanner, goal: frozenset[str]) -> Outcome:
        if self.model is None:
            self.model = env.catalog  # the agent's belief about the tools; refreshed only on evidence
        start_calls, harmful_before = env.calls, len(env.irreversible_log)
        state = env.snapshot()
        needed_irreversible = {
            n for n in self.model.necessary(goal, state) if n in self.model.tools and self.model.tools[n].irreversible
        }
        used_skill = False
        key = (goal, self.model.version)
        if self.skills and key in self.library:
            plan, corruption, used_skill = list(self.library[key]), None, True
        else:
            plan, corruption = planner.propose(self.model, goal, state)
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            if self.verify:
                plan = self._scope(plan, goal, state)
                for _ in range(6):  # repair in the sandbox, at no real cost
                    ok, error, reached = self._simulate(plan, state)
                    if ok and goal <= reached:
                        break
                    plan = self._patch(plan, error or "goal_unmet:", goal, state)
            snapshot = env.snapshot()
            error = None
            for name in plan:
                error = env.call(name)
                if error:
                    break
            if error is None and goal <= env.state:
                if self.skills:
                    self.library[(goal, self.model.version)] = list(plan)
                executed = env.irreversible_log[harmful_before:]
                # On success, harm is anything irreversible the goal did not need, or a needed action run twice.
                harmful = sum(1 for n in executed if n not in needed_irreversible)
                harmful += sum(max(0, executed.count(n) - 1) for n in needed_irreversible)
                return Outcome(True, env.calls - start_calls, harmful, attempts, used_skill, corruption, planner.calls)
            if error and error.startswith("precondition") and self.verify:
                name, _, missing = error.split(":", 1)[1].partition(":")
                if name in self.model.tools and not set(missing.split(",")) <= self.model.tools[name].requires:
                    self.model = learn_precondition(self.model, name, set(missing.split(",")))
                    self.learned += 1
            if error and error.startswith("unknown_tool") and "retired" in error:
                self.model = env.catalog  # refresh the tool catalog after a retirement notice
                if self.skills:
                    self.library = {k: v for k, v in self.library.items() if k[1] == self.model.version}
            env.rollback(snapshot)
            state = env.snapshot()
            if self.repair:
                plan = self._patch(plan, error or "goal_unmet:", goal, state)
            else:
                plan, _ = planner.propose(self.model, goal, state)
        # A failed task leaves every irreversible change it made behind.
        harmful = len(env.irreversible_log) - harmful_before
        return Outcome(False, env.calls - start_calls, harmful, attempts, used_skill, corruption, planner.calls)


def _distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def strategies() -> list[Agent]:
    return [
        Agent("single attempt"),
        Agent("blind retry (3 samples)", max_attempts=3),
        Agent("reflect and repair (3 attempts)", max_attempts=3, repair=True),
        Agent("repair in sandbox before acting", max_attempts=3, repair=True, verify=True),
        Agent("sandbox repair + verified skills", max_attempts=3, repair=True, verify=True, skills=True),
    ]
