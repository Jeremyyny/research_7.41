import unittest

from src.manager.marginal_value import (
    _make_sft_rows,
    choose_preferred_sequence,
    summarize_counterfactuals,
)


class PreferredSequenceTest(unittest.TestCase):
    def test_direct_correct_commits_even_when_calls_are_correct(self):
        branches = [
            {"sequence": ["reasoner"], "correct": True},
            {"sequence": ["extractor", "reasoner"], "correct": True},
        ]
        self.assertEqual(choose_preferred_sequence(True, branches), tuple())

    def test_shortest_success_is_selected(self):
        branches = [
            {"sequence": ["reasoner", "verifier"], "correct": True},
            {"sequence": ["verifier"], "correct": True},
            {"sequence": ["extractor"], "correct": False},
        ]
        self.assertEqual(choose_preferred_sequence(False, branches), ("verifier",))

    def test_unsolved_example_has_no_training_target(self):
        branches = [{"sequence": ["reasoner"], "correct": False}]
        self.assertIsNone(choose_preferred_sequence(False, branches))

    def test_seeded_tie_break_is_reproducible(self):
        branches = [
            {"sequence": ["reasoner"], "correct": True},
            {"sequence": ["extractor"], "correct": True},
        ]
        first = choose_preferred_sequence(False, branches, tie_break_seed=17)
        second = choose_preferred_sequence(False, branches, tie_break_seed=17)
        self.assertEqual(first, second)
        self.assertIn(first, {("reasoner",), ("extractor",)})


class CounterfactualSummaryTest(unittest.TestCase):
    def test_rescue_and_corruption_are_counted_from_paired_branches(self):
        records = [
            {
                "direct_valid": True,
                "direct_correct": False,
                "preferred_sequence": ["reasoner"],
                "branches": [
                    {"sequence": ["reasoner"], "correct": True},
                    {"sequence": ["extractor"], "correct": False},
                    {"sequence": ["verifier"], "correct": False},
                ],
            },
            {
                "direct_valid": True,
                "direct_correct": True,
                "preferred_sequence": [],
                "branches": [
                    {"sequence": ["reasoner"], "correct": False},
                    {"sequence": ["extractor"], "correct": True},
                    {"sequence": ["verifier"], "correct": True},
                ],
            },
        ]
        report = summarize_counterfactuals(records)
        self.assertEqual(report["direct_accuracy"], 0.5)
        self.assertEqual(report["oracle_accuracy"], 1.0)
        self.assertEqual(report["oracle_gain"], 0.5)
        self.assertEqual(report["by_advisor_one_step"]["reasoner"]["rescue_count"], 1)
        self.assertEqual(report["by_advisor_one_step"]["reasoner"]["corruption_count"], 1)


class SFTMaterializationTest(unittest.TestCase):
    def test_rescue_branch_becomes_call_then_commit_turns(self):
        tool_call = {
            "id": "mv_7_reasoner",
            "type": "function",
            "function": {"name": "reasoner_tool", "arguments": "{}"},
        }
        record = {
            "example_id": 7,
            "question_hash": "q7",
            "direct_pred": "A",
            "preferred_sequence": ["reasoner"],
            "base_messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
            "branches": [{
                "sequence": ["reasoner"],
                "final_pred": "B",
                "trajectory": [{
                    "role": "assistant",
                    "content": "DRAFT_ANSWER_A",
                    "tool_calls": [tool_call],
                    "tool_event": {
                        "role": "tool",
                        "tool_call_id": "mv_7_reasoner",
                        "name": "reasoner_tool",
                        "content": "evidence",
                    },
                }],
            }],
        }
        rows = _make_sft_rows(record)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["decision_type"], "call")
        self.assertEqual(rows[1]["decision_type"], "commit_after_call")
        self.assertEqual(rows[1]["response"][0]["content"], "DRAFT_ANSWER_B\nANSWER_B")
        self.assertEqual(rows[1]["prompt"][-1]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
