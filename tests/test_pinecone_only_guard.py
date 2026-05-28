import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.test_pinecone_vector_store import FakeIndex


class PineconeOnlyGuardTests(unittest.TestCase):
    def test_rejects_non_pinecone_vector_db_configuration(self):
        models = importlib.import_module("api.core.models")

        with (
            mock.patch.object(models, "_vector_collection", None),
            mock.patch.object(models, "VECTOR_DB", "local"),
            self.assertRaisesRegex(RuntimeError, "Pinecone"),
        ):
            models.get_vector_collection()

    def test_rejects_missing_pinecone_api_key(self):
        models = importlib.import_module("api.core.models")

        with (
            mock.patch.object(models, "_vector_collection", None),
            mock.patch.object(models, "VECTOR_DB", "pinecone"),
            mock.patch.object(models, "PINECONE_API_KEY", None),
            self.assertRaisesRegex(RuntimeError, "PINECONE_API_KEY"),
        ):
            models.get_vector_collection()

    def test_accepts_pinecone_configuration(self):
        fake_index = FakeIndex()

        class FakePinecone:
            def __init__(self, api_key):
                self.api_key = api_key

            def has_index(self, name):
                return True

            def describe_index(self, name):
                return {"status": {"ready": True}, "dimension": 2}

            def Index(self, name):
                return fake_index

        fake_module = types.ModuleType("pinecone")
        fake_module.Pinecone = FakePinecone
        fake_module.ServerlessSpec = lambda cloud, region: {"cloud": cloud, "region": region}

        with tempfile.TemporaryDirectory() as tmp_dir:
            models = importlib.import_module("api.core.models")
            with (
                mock.patch.dict(sys.modules, {"pinecone": fake_module}),
                mock.patch.object(models, "_vector_collection", None),
                mock.patch.object(models, "VECTOR_DB", "pinecone"),
                mock.patch.object(models, "PINECONE_API_KEY", "test-key"),
                mock.patch.object(models, "PINECONE_INDEX_NAME", "hansung-notices-test"),
                mock.patch.object(models, "PINECONE_NAMESPACE", "test"),
                mock.patch.object(models, "PINECONE_CACHE_PATH", str(Path(tmp_dir) / "chunks.json")),
                mock.patch.object(models, "EMBEDDING_DIM", 2),
            ):
                collection = models.get_vector_collection()

        self.assertIs(collection.index, fake_index)
        self.assertEqual(collection.namespace, "test")


if __name__ == "__main__":
    unittest.main()
