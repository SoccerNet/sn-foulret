import unittest

from retrieval.models import (
    MODEL_REGISTRY,
    model_requires_lora,
)


class FineTunedModelRegistryTests(unittest.TestCase):
    def test_paper_run_is_the_only_finetuned_key(self):
        finetuned = [key for key in MODEL_REGISTRY if "finetuned" in key]
        self.assertEqual(finetuned, ["qwen3vl-2b-finetuned"])
        self.assertTrue(model_requires_lora(finetuned[0]))
        self.assertFalse(model_requires_lora("qwen3vl-2b"))

    def test_internvideo_checkpoint_repositories(self):
        self.assertEqual(
            MODEL_REGISTRY["internvideo2-1b"]["hf_name"],
            "OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
        )
        self.assertEqual(
            MODEL_REGISTRY["internvideo2-6b"]["hf_name"],
            "OpenGVLab/InternVideo2-Stage2_6B-224p-f4",
        )


if __name__ == "__main__":
    unittest.main()
