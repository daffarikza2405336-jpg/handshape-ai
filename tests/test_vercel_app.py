from __future__ import annotations

import unittest

from api.index import app


class VercelAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {"status": "ok", "model_loaded": True},
        )

    def test_homepage_is_served(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"HandShape AI", response.data)


if __name__ == "__main__":
    unittest.main()
