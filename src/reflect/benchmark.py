"""A stream of tasks with drift half-way, replayed against every strategy with identical planner draws."""

from __future__ import annotations

import random
import statistics
from collections import defaultdict

from .agents import NoisyPlanner, Outcome, strategies
from .world import GOALS, Environment, base_catalog, degrade, drifted_catalog


def run_stream(
    agent, tasks: int = 500, drift_at: int | None = 250, error_rate: float = 0.35, seed: int = 0, model_gaps: int = 0
):
    goal_rng = random.Random(seed)
    goals = [goal_rng.choice(GOALS) for _ in range(tasks)]
    base, drifted = base_catalog(), drifted_catalog()
    if model_gaps:
        agent.model = degrade(base, model_gaps, random.Random(f"gaps:{seed}"))
    outcomes: list[Outcome] = []
    for index, goal in enumerate(goals):
        catalog = drifted if drift_at is not None and index >= drift_at else base
        env = Environment(catalog, set())
        planner = NoisyPlanner(error_rate, random.Random(f"{seed}:{index}"))
        outcomes.append(agent.solve(env, planner, goal))
    return outcomes


def summarize(outcomes: list[Outcome], drift_at: int | None) -> dict:
    n = len(outcomes)
    post = outcomes[drift_at : drift_at + 50] if drift_at is not None else []
    by_corruption = defaultdict(list)
    for o in outcomes:
        by_corruption[o.corruption or "none"].append(o.success)
    return {
        "success_rate": sum(o.success for o in outcomes) / n,
        "success_first_50_after_drift": sum(o.success for o in post) / len(post) if post else None,
        "real_calls_per_task": statistics.fmean(o.real_calls for o in outcomes),
        "real_calls_per_success": sum(o.real_calls for o in outcomes) / max(1, sum(o.success for o in outcomes)),
        "planner_calls_per_task": statistics.fmean(o.planner_calls for o in outcomes),
        "harmful_per_100_tasks": 100 * sum(o.harmful for o in outcomes) / n,
        "harmful_first_50_after_drift": sum(o.harmful for o in post) if post else None,
        "skill_reuse_rate": sum(o.used_skill for o in outcomes) / n,
        "success_by_planner_error": {k: sum(v) / len(v) for k, v in sorted(by_corruption.items())},
    }


def _fresh(template):
    return type(template)(**{k: getattr(template, k) for k in ("name", "max_attempts", "repair", "verify", "skills")})


def error_rate_sweep(rates=(0.0, 0.1, 0.2, 0.35, 0.5, 0.7), seeds: int = 3, tasks: int = 300) -> dict:
    """Success and harm as the planner gets worse (no drift, to isolate planner quality)."""
    out = {}
    for template in strategies():
        rows = []
        for rate in rates:
            runs = [summarize(run_stream(_fresh(template), tasks, None, rate, s), None) for s in range(seeds)]
            rows.append(
                {
                    "error_rate": rate,
                    "success_rate": statistics.fmean(r["success_rate"] for r in runs),
                    "harmful_per_100_tasks": statistics.fmean(r["harmful_per_100_tasks"] for r in runs),
                }
            )
        out[template.name] = rows
    return out


def model_fidelity(gaps=(0, 1, 2, 4, 6), seeds: int = 5, tasks: int = 500, error_rate: float = 0.35) -> dict:
    """The sandbox strategy with a world model missing some true preconditions.

    It learns each missing precondition the first time the real system reports it.
    """
    template = strategies()[3]
    rows = []
    for gap in gaps:
        runs, harm_windows, learned = [], [], []
        for seed in range(seeds):
            agent = _fresh(template)
            outcomes = run_stream(agent, tasks, None, error_rate, seed, model_gaps=gap)
            runs.append(summarize(outcomes, None))
            harm_windows.append([sum(o.harmful for o in outcomes[i : i + 50]) for i in range(0, tasks, 50)])
            learned.append(agent.learned)
        rows.append(
            {
                "missing_preconditions": gap,
                "success_rate": statistics.fmean(r["success_rate"] for r in runs),
                "harmful_per_100_tasks": statistics.fmean(r["harmful_per_100_tasks"] for r in runs),
                "preconditions_learned": statistics.fmean(learned),
                "harm_per_50_task_window": [
                    statistics.fmean(w[i] for w in harm_windows) for i in range(len(harm_windows[0]))
                ],
            }
        )
    return {"strategy": template.name, "rows": rows}


def compare(seeds: int = 5, tasks: int = 500, drift_at: int | None = 250, error_rate: float = 0.35) -> dict:
    rows = {}
    for template in strategies():
        per_seed = []
        for seed in range(seeds):
            agent = _fresh(template)
            per_seed.append(summarize(run_stream(agent, tasks, drift_at, error_rate, seed), drift_at))
        rows[template.name] = {
            key: (
                statistics.fmean(s[key] for s in per_seed)
                if not isinstance(per_seed[0][key], dict)
                else {k: statistics.fmean(s[key][k] for s in per_seed if k in s[key]) for k in per_seed[0][key]}
            )
            for key in per_seed[0]
        }
    return {
        "setup": {"tasks": tasks, "drift_at": drift_at, "planner_error_rate": error_rate, "seeds": seeds},
        "strategies": rows,
    }
