import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import asgi
from app.config import config


class TestAPIAuthenticationHTTP(unittest.TestCase):
    """Verify V1 API optional auth through the real ASGI app for two route groups."""

    def setUp(self):
        self.original_app_config = dict(config.app)
        self.client = TestClient(asgi.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_empty_key_preserves_existing_open_access(self):
        """The default empty key requires no header so old clients and the local WebUI work."""

        config.app["api_key"] = ""

        response = self.client.get("/api/v1/tasks")

        self.assertEqual(response.status_code, 200)

    def test_video_routes_require_matching_key_when_configured(self):
        """With protection enabled, video routes reject both missing and wrong keys."""

        config.app["api_key"] = "video-secret"

        missing = self.client.get("/api/v1/tasks")
        wrong = self.client.get(
            "/api/v1/tasks",
            headers={"x-api-key": "wrong"},
        )
        accepted = self.client.get(
            "/api/v1/tasks",
            headers={"x-api-key": "video-secret"},
        )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(accepted.status_code, 200)

    def test_llm_routes_authenticate_before_request_validation(self):
        """LLM routes authenticate first; unauthenticated calls never reach billable logic."""

        config.app["api_key"] = "llm-secret"

        # The request model supplies defaults, so even an empty request could
        # really call the LLM. Isolate the external service and check the call
        # count, verifying both auth order and that tests do not spend the
        # user's API quota.
        with patch(
            "app.controllers.v1.llm.llm.generate_script",
            return_value="mocked script",
        ) as generate_script:
            missing = self.client.post("/api/v1/scripts", json={})
            accepted = self.client.post(
                "/api/v1/scripts",
                json={},
                headers={"x-api-key": "llm-secret"},
            )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(accepted.status_code, 200)
        generate_script.assert_called_once()

    def test_openapi_documents_api_key_header_for_v1_routes(self):
        """Swagger must expose x-api-key instead of leaving the request format to guesswork."""

        schema = self.client.get("/openapi.json").json()
        parameters = schema["paths"]["/api/v1/tasks"]["get"]["parameters"]

        self.assertTrue(
            any(
                parameter["in"] == "header" and parameter["name"] == "x-api-key"
                for parameter in parameters
            )
        )

    def test_duplicate_api_key_headers_are_rejected(self):
        """Duplicate credentials may be interpreted differently per proxy, so reject any order."""

        config.app["api_key"] = "video-secret"

        correct_first = self.client.get(
            "/api/v1/tasks",
            headers=[
                ("x-api-key", "video-secret"),
                ("x-api-key", "wrong"),
            ],
        )
        wrong_first = self.client.get(
            "/api/v1/tasks",
            headers=[
                ("x-api-key", "wrong"),
                ("x-api-key", "video-secret"),
            ],
        )

        self.assertEqual(correct_first.status_code, 401)
        self.assertEqual(wrong_first.status_code, 401)


if __name__ == "__main__":
    unittest.main()
