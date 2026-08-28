import unittest
from unittest.mock import patch

from app.models.visual_scene import VisualScene
from app.services import llm, visual_scenes


VALID_SCENE_JSON = """
[
  {
    "id": 1,
    "narration": "This beach is completely black.",
    "visual_description": "dramatic black sand beach with ocean waves",
    "search_queries": [
      "black sand beach",
      "volcanic beach",
      "dark beach ocean"
    ]
  },
  {
    "id": 2,
    "narration": "The sand was created by volcanic rock.",
    "visual_description": "dark volcanic rocks on a coastline",
    "search_queries": [
      "volcanic rocks",
      "lava rocks beach",
      "black rocks ocean"
    ]
  }
]
""".strip()


class TestVisualSceneParsing(unittest.TestCase):
    def test_successful_parsing_returns_typed_scenes(self):
        scenes = visual_scenes.parse_visual_scenes(VALID_SCENE_JSON)

        self.assertEqual(len(scenes), 2)
        self.assertTrue(all(isinstance(scene, VisualScene) for scene in scenes))
        self.assertEqual(
            scenes[0].visual_description,
            "dramatic black sand beach with ocean waves",
        )

    def test_code_fenced_json_is_accepted(self):
        scenes = visual_scenes.parse_visual_scenes(f"```json\n{VALID_SCENE_JSON}\n```")

        self.assertEqual([scene.id for scene in scenes], [1, 2])

    def test_embedded_json_is_recovered(self):
        scenes = visual_scenes.parse_visual_scenes(
            f"Here is the requested plan:\n{VALID_SCENE_JSON}\nDone."
        )

        self.assertEqual(len(scenes), 2)

    def test_exactly_three_queries_are_required(self):
        invalid = VALID_SCENE_JSON.replace(
            '"volcanic beach",\n      "dark beach ocean"',
            '"volcanic beach"',
        )

        with self.assertRaisesRegex(ValueError, "exactly 3 queries"):
            visual_scenes.parse_visual_scenes(invalid)

    def test_duplicate_queries_are_rejected(self):
        invalid = VALID_SCENE_JSON.replace(
            '"volcanic beach",\n      "dark beach ocean"',
            '"black sand beach",\n      "dark beach ocean"',
        )

        with self.assertRaisesRegex(ValueError, "distinct"):
            visual_scenes.parse_visual_scenes(invalid)

    def test_chronological_order_is_preserved_and_validated(self):
        scenes = visual_scenes.parse_visual_scenes(VALID_SCENE_JSON)

        self.assertEqual(
            [scene.narration for scene in scenes],
            [
                "This beach is completely black.",
                "The sand was created by volcanic rock.",
            ],
        )

        reversed_ids = VALID_SCENE_JSON.replace('"id": 1', '"id": 9', 1)
        with self.assertRaisesRegex(ValueError, "chronological"):
            visual_scenes.parse_visual_scenes(reversed_ids)


class TestGenerateVisualScenes(unittest.TestCase):
    def test_malformed_response_retries_with_next_response(self):
        with patch.object(
            llm,
            "_generate_response",
            side_effect=["[{not valid json]", VALID_SCENE_JSON],
        ) as generate:
            scenes = llm.generate_visual_scenes(
                video_subject="black sand beaches",
                video_script="A black beach formed from volcanic rock.",
            )

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(len(scenes), 2)

    def test_empty_llm_response_returns_empty_list_after_retries(self):
        with patch.object(llm, "_generate_response", return_value="") as generate:
            scenes = llm.generate_visual_scenes(
                video_subject="owls",
                video_script="An owl flies through a forest.",
            )

        self.assertEqual(scenes, [])
        self.assertEqual(generate.call_count, llm._max_retries)

    def test_prompt_requests_stock_footage_constraints(self):
        captured = {}

        def fake_generate_response(prompt):
            captured["prompt"] = prompt
            return VALID_SCENE_JSON

        with patch.object(
            llm, "_generate_response", side_effect=fake_generate_response
        ):
            scenes = llm.generate_visual_scenes(
                video_subject="black sand beaches",
                video_script="A black beach formed from volcanic rock.",
                target_scene_count=5,
            )

        self.assertEqual(len(scenes), 2)
        self.assertIn("exactly 3 distinct English search_queries", captured["prompt"])
        self.assertIn("roughly every 3-5 seconds", captured["prompt"])
        self.assertIn("approximately 5 scenes", captured["prompt"])
        self.assertIn("visible nouns and actions", captured["prompt"])

    def test_generate_terms_contract_is_unchanged(self):
        with patch.object(
            llm,
            "_generate_response",
            return_value='["black beach", "volcanic rocks"]',
        ):
            terms = llm.generate_terms(
                video_subject="black sand beaches",
                video_script="A black beach formed from volcanic rock.",
                amount=2,
            )

        self.assertEqual(terms, ["black beach", "volcanic rocks"])
        self.assertTrue(all(isinstance(term, str) for term in terms))


if __name__ == "__main__":
    unittest.main()
