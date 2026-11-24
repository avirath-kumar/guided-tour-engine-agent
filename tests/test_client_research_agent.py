import unittest
from unittest.mock import patch

from agents.client_research_agent import CLIENTS, _client_lookup_error, get_client_profile


class ClientProfileTests(unittest.TestCase):
    def test_crm_id_preserves_profile(self):
        self.assertEqual(
            get_client_profile.invoke({"client_id": " cl-0042 "}),
            "**Acme Pension Fund** (Institutional — pension)\n"
            "Account: ACCT-00420042\n"
            "AUM: $1,200,000,000\n"
            "Risk profile: Moderate, long-duration\n"
            "Last meeting: 2026-02-14\n"
            "Top holdings:\n  - AAPL: 8.0%\n  - MSFT: 7.0%\n  - JPM: 5.0%\n  - XOM: 4.0%",
        )

    def test_names_return_existing_profiles(self):
        cases = [
            ("Acme Pension Fund", "CL-0042"),
            (" sMiTh FAMILY office ", "CL-0117"),
            ("SMITH", "CL-0117"),
            ("Smith-Family_Office!", "CL-0117"),
            ("office smith", "CL-0117"),
            ("Delgado", "CL-0405"),
            ("TechCorp Treasury", "CL-0263"),
            ("Harbor University Endowment", "CL-0318"),
        ]
        for name, crm_id in cases:
            with self.subTest(name=name):
                result = get_client_profile.invoke({"client_id": name})
                expected = get_client_profile.invoke({"client_id": crm_id})
                header, body = expected.split("\n", 1)
                self.assertEqual(result, f"{header} [{crm_id}]\n{body}")

    def test_misses_ask_for_clarification(self):
        for name in ("Unknown Fund", "", "---", "Smi", "Smith Unknown"):
            with self.subTest(name=name), self.assertRaises(ValueError) as error:
                get_client_profile.invoke({"client_id": name})
            message = _client_lookup_error(error.exception, None)
            self.assertIn("No client found", message)
            self.assertIn("CL-0042", message)
            self.assertIn("full client name", message)
            self.assertIn("do not guess slugs", message)

    def test_ambiguous_name_lists_candidates(self):
        with self.assertRaises(ValueError) as error:
            get_client_profile.invoke({"client_id": "Family"})
        message = _client_lookup_error(error.exception, None)
        for text in ("Ambiguous", "Smith Family Office", "Delgado Family Trust", "CL-0042", "full client name"):
            self.assertIn(text, message)

    def test_exact_name_precedes_partial_matches(self):
        with patch.dict(CLIENTS, {"CL-9999": {**CLIENTS["CL-0117"], "name": "Smith Family Office Trust"}}):
            self.assertIn("[CL-0117]", get_client_profile.invoke({"client_id": "Smith Family Office"}))

    def test_duplicate_exact_names_are_ambiguous(self):
        with patch.dict(CLIENTS, {"CL-9999": {**CLIENTS["CL-0117"]}}):
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                get_client_profile.invoke({"client_id": "Smith Family Office"})


if __name__ == "__main__":
    unittest.main()
