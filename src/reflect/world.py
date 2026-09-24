"""A release-engineering tool world with irreversible actions, and the tasks agents must solve.

Tools have preconditions and effects over a set of facts. Some tools are
irreversible (migrating a database, deploying, rotating keys, announcing):
if an attempt runs one and then fails, that change stays behind. Mid-stream,
the world can *drift*: ``deploy`` is retired in favour of ``deploy_v2``, which
additionally requires a canary run, so plans and skills learned earlier break.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    requires: frozenset[str]
    adds: frozenset[str]
    irreversible: bool = False


def _tool(name: str, requires: str, adds: str, irreversible: bool = False) -> Tool:
    return Tool(name, frozenset(requires.split()), frozenset(adds.split()), irreversible)


BASE_TOOLS = (
    _tool("open_ticket", "", "ticket"),
    _tool("get_approval", "ticket", "approved"),
    _tool("snapshot_db", "ticket", "backup"),
    _tool("run_tests", "ticket", "tests_passed"),
    _tool("build_artifact", "tests_passed", "artifact"),
    _tool("stage_release", "artifact", "staged"),
    _tool("scale_up", "ticket", "capacity"),
    _tool("migrate_db", "approved backup", "migrated", irreversible=True),
    _tool("deploy", "staged approved", "deployed", irreversible=True),
    _tool("rotate_keys", "approved backup", "keys_rotated", irreversible=True),
    _tool("announce", "deployed", "announced", irreversible=True),
    _tool("purge_cache", "deployed capacity", "cache_purged", irreversible=True),
)
DRIFT_TOOLS = (
    _tool("run_canary", "staged", "canary_passed"),
    _tool("deploy_v2", "staged approved canary_passed", "deployed", irreversible=True),
)

GOALS = (
    frozenset({"deployed"}),
    frozenset({"deployed", "migrated"}),
    frozenset({"keys_rotated"}),
    frozenset({"deployed", "announced"}),
    frozenset({"cache_purged"}),
    frozenset({"migrated"}),
    frozenset({"deployed", "keys_rotated"}),
    frozenset({"announced", "cache_purged"}),
)


class Catalog:
    """The set of tools a world exposes, with a version that changes on drift."""

    def __init__(self, tools: tuple[Tool, ...], version: int) -> None:
        self.tools = {t.name: t for t in tools}
        self.version = version

    def producer(self, fact: str) -> Tool | None:
        return next((t for t in self.tools.values() if fact in t.adds), None)

    def plan(self, goal: frozenset[str], state: frozenset[str] = frozenset()) -> list[str]:
        """A minimal ordered plan by backward chaining (each fact has one producer)."""
        ordered: list[str] = []

        def achieve(fact: str, depth: int = 0) -> None:
            if fact in state or depth > 20:
                return
            tool = self.producer(fact)
            if tool is None:
                raise ValueError(f"no tool produces {fact!r}")
            for needed in sorted(tool.requires):
                achieve(needed, depth + 1)
            if tool.name not in ordered:
                ordered.append(tool.name)

        for fact in sorted(goal):
            achieve(fact)
        return ordered

    def necessary(self, goal: frozenset[str], state: frozenset[str] = frozenset()) -> set[str]:
        return set(self.plan(goal, state))


def degrade(catalog: Catalog, gaps: int, rng) -> Catalog:
    """A copy of ``catalog`` whose tools are missing ``gaps`` of their true preconditions.

    This is the agent's imperfect world model: a sandbox built from it will
    approve plans that the real system rejects.
    """
    pairs = sorted((t.name, fact) for t in catalog.tools.values() for fact in t.requires)
    removed = set(rng.sample(pairs, k=min(gaps, len(pairs))))
    tools = tuple(
        Tool(t.name, frozenset(f for f in t.requires if (t.name, f) not in removed), t.adds, t.irreversible)
        for t in catalog.tools.values()
    )
    return Catalog(tools, catalog.version)


def learn_precondition(catalog: Catalog, tool_name: str, facts: set[str]) -> Catalog:
    """Return a model that also knows ``tool_name`` requires ``facts`` (learned from a real error)."""
    tools = tuple(
        Tool(t.name, t.requires | frozenset(facts), t.adds, t.irreversible) if t.name == tool_name else t
        for t in catalog.tools.values()
    )
    return Catalog(tools, catalog.version)


def base_catalog() -> Catalog:
    return Catalog(BASE_TOOLS, version=1)


def drifted_catalog() -> Catalog:
    tools = tuple(t for t in BASE_TOOLS if t.name != "deploy") + DRIFT_TOOLS
    return Catalog(tools, version=2)


@dataclass
class Environment:
    """The real system. Tool calls cost, and irreversible effects cannot be undone."""

    catalog: Catalog
    state: set[str]
    calls: int = 0
    irreversible_log: list[str] | None = None

    def __post_init__(self) -> None:
        self.irreversible_log = []

    def call(self, name: str) -> str | None:
        """Run one tool. Returns an error message, or None on success."""
        self.calls += 1
        tool = self.catalog.tools.get(name)
        if tool is None:
            hint = " (retired; see catalog)" if name == "deploy" and self.catalog.version > 1 else ""
            return f"unknown_tool:{name}{hint}"
        missing = tool.requires - self.state
        if missing:
            return f"precondition:{name}:{','.join(sorted(missing))}"
        self.state |= tool.adds
        if tool.irreversible:
            self.irreversible_log.append(name)
        return None

    def snapshot(self) -> frozenset[str]:
        return frozenset(self.state)

    def rollback(self, snapshot: frozenset[str]) -> None:
        """Undo reversible effects since ``snapshot``; irreversible facts stay."""
        irreversible_facts = {f for t in self.catalog.tools.values() if t.irreversible for f in t.adds}
        self.state = set(snapshot) | (self.state & irreversible_facts)
