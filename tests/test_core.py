import unittest

from reflective_agent_lab.core import Environment, ReflectiveAgent, Task, Tool, fixture


class AgentTests(unittest.TestCase):
    def test_repairs_missing_verification(self):
        agent, tasks = fixture()
        result = agent.solve(*tasks[0])
        self.assertTrue(result["trace"].success)
        self.assertIn("check", result["trace"].actions)

    def test_repairs_ordering(self):
        agent, tasks = fixture()
        result = agent.solve(*tasks[1])
        self.assertEqual(result["trace"].actions[:3], ["observe", "authenticate", "change"])

    def test_does_not_reflect_success(self):
        agent, tasks = fixture()
        result = agent.solve(*tasks[2])
        self.assertFalse(result["reflected"])

    def test_unavailable_tool_is_not_invented(self):
        env = Environment([Tool("observe", frozenset(), frozenset({"seen"}))])
        agent = ReflectiveAgent(env)
        task = Task("impossible", frozenset(), frozenset({"done"}), ("observe",))
        result = agent.solve(task, ["observe"])
        self.assertFalse(result["trace"].success)
        self.assertNotIn("impossible_tool", result["trace"].actions)


if __name__ == "__main__":
    unittest.main()
