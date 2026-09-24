import random

import pytest

from reflect.agents import Agent, NoisyPlanner, strategies
from reflect.benchmark import compare, model_fidelity, run_stream, summarize
from reflect.world import GOALS, Environment, base_catalog, degrade, drifted_catalog


def test_catalog_plans_reach_every_goal_in_both_worlds():
    for catalog in (base_catalog(), drifted_catalog()):
        for goal in GOALS:
            env = Environment(catalog, set())
            for name in catalog.plan(goal):
                assert env.call(name) is None
            assert goal <= env.state


def test_environment_errors_and_irreversibility():
    env = Environment(base_catalog(), set())
    assert env.call("deploy").startswith("precondition:deploy:")
    assert env.call("teleport") == "unknown_tool:teleport"
    snap = env.snapshot()
    for name in ("open_ticket", "get_approval", "snapshot_db", "migrate_db"):
        assert env.call(name) is None
    env.rollback(snap)
    assert env.state == {"migrated"}  # reversible effects undone, the migration is not
    assert env.irreversible_log == ["migrate_db"]


def test_retired_tool_says_so():
    env = Environment(drifted_catalog(), {"staged", "approved"})
    assert "retired" in env.call("deploy")


@pytest.mark.parametrize("kind", NoisyPlanner.KINDS)
def test_each_corruption_breaks_a_plan_or_widens_its_scope(kind):
    catalog, goal = base_catalog(), frozenset({"deployed", "migrated"})
    for seed in range(200):
        planner = NoisyPlanner(1.0, random.Random(seed))
        plan, got = planner.propose(catalog, goal, frozenset())
        if got != kind:
            continue
        env = Environment(catalog, set())
        errors = [e for e in (env.call(n) for n in plan) if e]
        assert errors or len(env.irreversible_log) > 2
        return
    pytest.fail(f"no {kind} corruption sampled")


def test_sandbox_agent_removes_unrequested_irreversible_actions():
    agent = Agent("s", max_attempts=3, repair=True, verify=True)
    catalog = base_catalog()
    plan = ["open_ticket", "rotate_keys", "get_approval"]
    agent.model = catalog
    assert agent._scope(plan, frozenset({"approved"}), frozenset()) == ["open_ticket", "get_approval"]


def test_repair_moves_a_late_producer_forward():
    agent = Agent("r", repair=True)
    agent.model = base_catalog()
    plan = ["open_ticket", "build_artifact", "run_tests"]
    fixed = agent._patch(plan, "precondition:build_artifact:tests_passed", frozenset({"artifact"}), frozenset())
    assert fixed == ["open_ticket", "run_tests", "build_artifact"]


def test_skills_are_reused_and_invalidated_by_drift():
    agent = strategies()[4]
    outcomes = run_stream(agent, tasks=120, drift_at=60, error_rate=0.35, seed=1)
    assert all(o.success for o in outcomes)
    assert sum(o.used_skill for o in outcomes[:60]) > 40
    assert all(key[1] == 2 for key in agent.library)  # only post-drift skills survive


def test_degraded_model_is_learned_back():
    agent = strategies()[3]
    rng = random.Random(0)
    agent.model = degrade(base_catalog(), 3, rng)
    missing = sum(len(t.requires) for t in base_catalog().tools.values()) - sum(
        len(t.requires) for t in agent.model.tools.values()
    )
    assert missing == 3
    outcomes = run_stream(agent, tasks=200, drift_at=None, error_rate=0.35, seed=0)
    assert sum(o.success for o in outcomes) >= 198
    assert agent.learned >= 1


def test_headline_ordering_holds():
    report = compare(seeds=2, tasks=200, drift_at=100)["strategies"]
    names = [a.name for a in strategies()]
    single, blind, reflect, sandbox, skills = (report[n] for n in names)
    assert blind["success_rate"] > single["success_rate"]
    assert reflect["harmful_per_100_tasks"] > blind["harmful_per_100_tasks"]  # acting to learn is costly
    assert sandbox["harmful_per_100_tasks"] < 1 and sandbox["success_rate"] > 0.99
    assert skills["planner_calls_per_task"] < 0.2 < sandbox["planner_calls_per_task"]


def test_model_fidelity_rows():
    rows = model_fidelity(gaps=(0, 2), seeds=1, tasks=100)["rows"]
    assert rows[0]["harmful_per_100_tasks"] == 0 and rows[1]["preconditions_learned"] >= 1


def test_summary_fields():
    summary = summarize(run_stream(strategies()[0], tasks=50, drift_at=None), None)
    assert 0 <= summary["success_rate"] <= 1 and summary["success_first_50_after_drift"] is None
