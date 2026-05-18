import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from api.core.models import PineconeCollectionAdapter


class FakeIndex:
    def __init__(self):
        self.vectors = {}

    def upsert(self, vectors, namespace):
        for vector in vectors:
            self.vectors[vector["id"]] = vector

    def fetch(self, ids, namespace):
        return SimpleNamespace(
            vectors={id_: self.vectors[id_] for id_ in ids if id_ in self.vectors}
        )

    def delete(self, ids, namespace):
        for id_ in ids:
            self.vectors.pop(id_, None)

    def query(self, vector, top_k, namespace, filter=None, include_metadata=True):
        matches = []
        for id_, stored in self.vectors.items():
            metadata = stored["metadata"]
            if filter and any(metadata.get(key) != value for key, value in filter.items()):
                continue
            score = sum(a * b for a, b in zip(vector, stored["values"]))
            matches.append(SimpleNamespace(id=id_, score=score, metadata=metadata))
        matches.sort(key=lambda match: match.score, reverse=True)
        return SimpleNamespace(matches=matches[:top_k])

    def describe_index_stats(self):
        return {"namespaces": {"test": {"vector_count": len(self.vectors)}}}


class PineconeVectorStoreTests(unittest.TestCase):
    def test_pinecone_adapter_preserves_chroma_like_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = PineconeCollectionAdapter(
                index=FakeIndex(),
                namespace="test",
                cache_path=str(Path(tmp_dir) / "chunks.json"),
            )

            adapter.add(
                ids=["notice-1_0", "notice-2_0"],
                embeddings=[[1.0, 0.0], [0.0, 1.0]],
                documents=["장학금 신청 안내", "인턴십 모집 안내"],
                metadatas=[
                    {"title": "장학금", "url": "https://example.com/1", "category": "장학금"},
                    {"title": "인턴십", "url": "https://example.com/2", "category": "인턴십"},
                ],
            )

            all_data = adapter.get(include=["documents", "metadatas"])
            self.assertEqual(all_data["ids"], ["notice-1_0", "notice-2_0"])
            self.assertEqual(all_data["documents"], ["장학금 신청 안내", "인턴십 모집 안내"])
            self.assertEqual(
                all_data["metadatas"][0],
                {
                    "title": "장학금",
                    "url": "https://example.com/1",
                    "category": "장학금",
                },
            )

            filtered = adapter.get(include=["documents"], where={"category": "인턴십"})
            self.assertEqual(filtered["ids"], ["notice-2_0"])
            self.assertEqual(filtered["documents"], ["인턴십 모집 안내"])

            fetched = adapter.get(ids=["notice-1_0", "missing"], include=["documents", "metadatas"])
            self.assertEqual(fetched["ids"], ["notice-1_0"])
            self.assertEqual(fetched["documents"], ["장학금 신청 안내"])
            self.assertNotIn("document", fetched["metadatas"][0])

            queried = adapter.query(
                query_embeddings=[[0.9, 0.1]],
                n_results=1,
                include=["metadatas", "distances", "documents"],
            )
            self.assertEqual(queried["ids"], [["notice-1_0"]])
            self.assertEqual(
                queried["metadatas"],
                [[{
                    "title": "장학금",
                    "url": "https://example.com/1",
                    "category": "장학금",
                }]],
            )
            self.assertEqual(queried["documents"], [["장학금 신청 안내"]])
            self.assertAlmostEqual(queried["distances"][0][0], 0.1)

            adapter.delete(ids=["notice-1_0"])
            self.assertEqual(adapter.get(ids=["notice-1_0"])["ids"], [])
            self.assertEqual(adapter.count(), 1)

    def test_get_chroma_can_initialize_pinecone_adapter(self):
        fake_index = FakeIndex()

        class FakePinecone:
            def __init__(self, api_key):
                self.api_key = api_key
                self.created = []

            def has_index(self, name):
                return False

            def create_index(self, **kwargs):
                self.created.append(kwargs)

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
                mock.patch.object(models, "_chroma", None),
                mock.patch.object(models, "VECTOR_DB", "pinecone"),
                mock.patch.object(models, "PINECONE_API_KEY", "test-key"),
                mock.patch.object(models, "PINECONE_INDEX_NAME", "hansung-notices-test"),
                mock.patch.object(models, "PINECONE_NAMESPACE", "test"),
                mock.patch.object(models, "PINECONE_CACHE_PATH", str(Path(tmp_dir) / "chunks.json")),
                mock.patch.object(models, "EMBEDDING_DIM", 2),
            ):
                collection = models.get_chroma()

        self.assertIsInstance(collection, PineconeCollectionAdapter)
        self.assertIs(collection.index, fake_index)
        self.assertEqual(collection.namespace, "test")


if __name__ == "__main__":
    unittest.main()
