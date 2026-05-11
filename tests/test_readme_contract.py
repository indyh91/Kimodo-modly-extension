import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def readme_text():
    return README.read_text(encoding="utf-8")


class ReadmeContractTests(unittest.TestCase):
    def test_windows_x86_64_is_documented_as_experimental_not_validated_support(self):
        text = readme_text()
        lower_text = text.lower()

        self.assertIn("windows x86_64", lower_text)
        self.assertIn("experimental compatibility lane", lower_text)
        self.assertIn("has not yet been validated", lower_text)
        self.assertRegex(lower_text, r"cuda.*pytorch.*dll.*gated")

        forbidden_claims = (
            "windows x86_64 is supported",
            "validated windows support",
            "fully tested windows",
            "production windows support",
        )
        for claim in forbidden_claims:
            self.assertNotIn(claim, lower_text)

    def test_encoder_gated_access_note_names_required_models_and_access_conditions(self):
        text = readme_text()

        required_models = (
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        )
        for model_id in required_models:
            self.assertIn(model_id, text)

        self.assertRegex(text, r"(?is)Hugging Face.*Meta.*(?:accept|request).*access conditions.*before.*download")

    def test_alternative_encoders_are_framed_as_future_research_only(self):
        text = readme_text()
        lower_text = text.lower()

        self.assertRegex(lower_text, r"future research:.*alternative text encoders")
        self.assertRegex(lower_text, r"future research:.*external encoder service")
        self.assertNotRegex(lower_text, r"(?<!future research: )implemented alternative (?:text )?encoder")
        self.assertNotRegex(lower_text, r"supports alternative (?:text )?encoders")


if __name__ == "__main__":
    unittest.main()
