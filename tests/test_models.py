import traceback
import unittest
from unittest.mock import Mock, patch

from langsmith import traceable, tracing_context

from utils import models


class MakeModelTests(unittest.TestCase):
    def test_valid_name_with_key_builds_model(self):
        with (
            patch.object(models, "API_KEY_ENV", "TEST_MODEL_KEY_2"),
            patch.dict(models.os.environ, {"TEST_MODEL_KEY_2": "test-key"}, clear=True),
            patch.object(models, "init_chat_model") as init_model,
        ):
            result = models.make_model(max_retries=1)

        self.assertIs(result, init_model.return_value)
        init_model.assert_called_once_with(
            **models.MODEL_SPEC, max_retries=1, api_key="test-key"
        )

    def test_valid_name_without_key_keeps_missing_key_error(self):
        with (
            patch.object(models, "API_KEY_ENV", "TEST_MODEL_KEY_2"),
            patch.dict(models.os.environ, {}, clear=True),
            self.assertRaises(RuntimeError) as caught,
        ):
            models.make_model()

        self.assertEqual(
            str(caught.exception),
            "Set TEST_MODEL_KEY_2 in .env (see utils/models.py for the model in use).",
        )

    def test_invalid_names_do_not_leak_into_exceptions_or_traces(self):
        for setting in ("lsv2_pt_fake-test-credential", "lowercase_key", "BAD-KEY", "KEY\n", ""):
            with self.subTest(setting=setting):
                runs = []
                with (
                    patch.object(models, "API_KEY_ENV", setting),
                    patch.object(models.os, "getenv", wraps=models.os.getenv) as getenv,
                    patch.object(models, "init_chat_model") as init_model,
                    tracing_context(enabled=True, client=Mock()),
                    self.assertRaises(RuntimeError) as caught,
                ):
                    traceable(models.make_model)(langsmith_extra={"on_end": runs.append})

                self.assertEqual(
                    str(caught.exception),
                    "API_KEY_ENV in utils/models.py must be the name of an environment variable "
                    "(for example LANGSMITH_GATEWAY_API_KEY), not the API key itself. "
                    "Put the key in .env under that name.",
                )
                self.assertNotIn(setting, [call.args[0] for call in getenv.call_args_list])
                init_model.assert_not_called()
                self.assertEqual(len(runs), 1)
                self.assertIn(str(caught.exception), runs[0].error)
                if setting:
                    self.assertNotIn(setting, runs[0].error)
                    self.assertNotIn(setting, "".join(traceback.format_exception(caught.exception)))

    def test_none_uses_credential_chain(self):
        with (
            patch.object(models, "API_KEY_ENV", None),
            patch.object(models, "init_chat_model") as init_model,
        ):
            result = models.make_model()

        self.assertIs(result, init_model.return_value)
        init_model.assert_called_once_with(**models.MODEL_SPEC)
