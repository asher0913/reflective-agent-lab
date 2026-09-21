from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from statistics import mean


@dataclass(frozen=True)
class Tool:
    name: str
    requires: frozenset[str]
    adds: frozenset[str]


@dataclass(frozen=True)
class Task:
    name: str
    initial: frozenset[str]
    goal: frozenset[str]
    allowed: tuple[str, ...]


@dataclass
class Trace:
    actions: list[str]
    states: list[set[str]] = field(default_factory=list)
    error: str | None = None
    success: bool = False


class Environment:
    def __init__(self, tools: list[Tool]) -> None:
        self.tools = {tool.name: tool for tool in tools}

    def execute(self, task: Task, actions: list[str]) -> Trace:
        state = set(task.initial)
        trace = Trace(list(actions), [set(state)])
        for action in actions:
            if action not in task.allowed or action not in self.tools:
                trace.error = f"tool_gap:{action}"
                return trace
            tool = self.tools[action]
            missing = tool.requires - state
            if missing:
                trace.error = f"ordering_gap:{action}:{','.join(sorted(missing))}"
                return trace
            state.update(tool.adds)
            trace.states.append(set(state))
        trace.success = task.goal <= state
        if not trace.success:
            trace.error = "observation_gap:" + ",".join(sorted(task.goal - state))
        return trace


class ReflectiveAgent:
    def __init__(self, environment: Environment) -> None:
        self.environment = environment
        self.skills: dict[str, list[str]] = {}

    def verify(self, task: Task, trace: Trace) -> bool:
        return bool(trace.states) and task.goal <= trace.states[-1] and trace.error is None

    def diagnose(self, task: Task, trace: Trace) -> tuple[str, str | None]:
        if trace.error and trace.error.startswith("ordering_gap"):
            _, action, missing = trace.error.split(":", 2)
            need = set(missing.split(","))
            for tool in self.environment.tools.values():
                if need <= tool.adds and tool.name in task.allowed:
                    return "ordering_gap", tool.name
            return "tool_gap", None
        if trace.error and trace.error.startswith("tool_gap"):
            return "tool_gap", None
        missing = task.goal - (trace.states[-1] if trace.states else set())
        for tool in self.environment.tools.values():
            if missing & tool.adds and tool.name in task.allowed:
                return "observation_gap", tool.name
        return "unrecoverable", None

    def solve(self, task: Task, candidate: list[str]) -> dict[str, object]:
        baseline = self.environment.execute(task, candidate)
        if self.verify(task, baseline):
            self.skills[task.name] = list(candidate)
            return {"trace": baseline, "reflected": False, "rolled_back": False, "category": "none"}
        category, repair = self.diagnose(task, baseline)
        if repair is None:
            return {"trace": baseline, "reflected": True, "rolled_back": False, "category": category}
        repaired_actions = list(candidate)
        if category == "ordering_gap" and baseline.error:
            failed = baseline.error.split(":", 2)[1]
            repaired_actions.insert(max(0, repaired_actions.index(failed)), repair)
        else:
            repaired_actions.append(repair)
        repaired = self.environment.execute(task, repaired_actions)
        if not self.verify(task, repaired):
            return {"trace": baseline, "reflected": True, "rolled_back": True, "category": category}
        self.skills[task.name] = repaired_actions
        return {"trace": repaired, "reflected": True, "rolled_back": False, "category": category}


def fixture() -> tuple[ReflectiveAgent, list[tuple[Task, list[str]]]]:
    tools = [
        Tool("observe", frozenset(), frozenset({"observed"})),
        Tool("authenticate", frozenset({"observed"}), frozenset({"authorized"})),
        Tool("change", frozenset({"authorized"}), frozenset({"changed"})),
        Tool("check", frozenset({"changed"}), frozenset({"verified"})),
    ]
    agent = ReflectiveAgent(Environment(tools))
    tasks = [
        (Task("safe_change", frozenset(), frozenset({"verified"}), tuple(t.name for t in tools)), ["observe", "authenticate", "change"]),
        (Task("ordered_change", frozenset(), frozenset({"changed"}), tuple(t.name for t in tools)), ["observe", "change"]),
        (Task("already_good", frozenset(), frozenset({"authorized"}), tuple(t.name for t in tools)), ["observe", "authenticate"]),
    ]
    return agent, tasks


def demo() -> dict[str, object]:
    agent, tasks = fixture()
    outcomes = [agent.solve(task, actions) for task, actions in tasks]
    traces = [outcome["trace"] for outcome in outcomes]
    baseline_success = sum(1 for task, actions in tasks if agent.environment.execute(task, actions).success)
    return {
        "tasks": len(tasks),
        "baseline_pass_rate": baseline_success / len(tasks),
        "final_pass_rate": sum(1 for trace in traces if trace.success) / len(tasks),
        "mean_tool_calls": mean(len(trace.actions) for trace in traces),
        "rollbacks": sum(bool(outcome["rolled_back"]) for outcome in outcomes),
        "skills": agent.skills,
        "categories": [outcome["category"] for outcome in outcomes],
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, sort_keys=True))
