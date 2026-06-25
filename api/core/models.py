"""
Lazy-initialized singletons for heavy resources (models, DB).
Replaces @st.cache_resource from app.py.
"""
import hashlib
import gc
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

from .config import (
    EMBED_MODEL_PATH, BASE_MODEL_EMBED, EMBEDDER_BACKEND, SIMCSE_POOLING,
    SUMMARY_MODEL_PATH, CLASSIFY_MODEL_PATH,
    INDEX_MANIFEST_PATH, NOTICES_CACHE_PATH,
    VECTOR_DB, PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_CLOUD,
    PINECONE_REGION, PINECONE_NAMESPACE, PINECONE_CACHE_PATH, EMBEDDING_DIM,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_DEVICE,
)
from .utils import infer_category, chunk_text

_embed_model      = None
_summary_pipeline = None
_classifier       = None
_label_map: dict  = {}
_vector_collection = None

EMBEDDING_PIPELINE_VERSION = (
    f"simcse-{SIMCSE_POOLING}-v1"
    if EMBEDDER_BACKEND == "simcse"
    else "sentence-transformers-default-v1"
)
TEXT_PROCESSING_VERSION = "ko-compound-category-v2"


class SimCSEEmbedder:
    """
    SimCSE-aware embedder with selectable pooling.

    CLS pooling follows the common SimCSE inference path. Mean pooling is useful
    as an ablation for longer retrieval chunks where token-level evidence matters.
    """

    def __init__(self, model_path: str, device: str = "cpu", pooling: str = SIMCSE_POOLING) -> None:
        from transformers import AutoTokenizer, AutoModel

        pooling = pooling.lower()
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling: {pooling}. Use 'cls' or 'mean'.")

        local_only = os.getenv("TRANSFORMERS_OFFLINE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1"
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
        self._model = AutoModel.from_pretrained(model_path, local_files_only=local_only)
        self._model.eval()
        self._device = device
        self._pooling = pooling
        self._model.to(device)

    def encode(
        self,
        sentences: "str | list[str]",
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> "np.ndarray":
        import numpy as np
        import torch
        import torch.nn.functional as F

        single = isinstance(sentences, str)
        if single:
            sentences = [sentences]

        all_embeddings: list = []
        total = len(sentences)
        total_batches = (total + batch_size - 1) // batch_size
        started_at = time.monotonic()
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            batch_no = start // batch_size + 1
            if show_progress_bar:
                logger.info(
                    "청크 인코딩 진행 중: %d/%d 배치 (%d/%d 청크)",
                    batch_no,
                    total_batches,
                    min(start + len(batch), total),
                    total,
                )
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                output = self._model(**encoded)
            hidden = output.last_hidden_state
            if self._pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.cpu().numpy())
            if show_progress_bar:
                elapsed = time.monotonic() - started_at
                logger.info(
                    "청크 인코딩 완료: %d/%d 배치 (%.1f%%, %.1fs 경과)",
                    batch_no,
                    total_batches,
                    batch_no / total_batches * 100,
                    elapsed,
                )

        result = np.concatenate(all_embeddings, axis=0)
        return result[0] if single else result


def _release_torch_cache() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        logger.debug("Torch cache release skipped", exc_info=True)


def _best_device() -> str:
    import torch
    if EMBEDDING_DEVICE:
        return EMBEDDING_DEVICE
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        if EMBEDDER_BACKEND == "simcse":
            _embed_model = SimCSEEmbedder(_embed_model_source(), device=_best_device())
        elif EMBEDDER_BACKEND in {"sentence-transformers", "sentence_transformers", "st"}:
            from sentence_transformers import SentenceTransformer

            local_only = os.getenv("TRANSFORMERS_OFFLINE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1"
            _embed_model = SentenceTransformer(
                _embed_model_source(),
                device=_best_device(),
                local_files_only=local_only,
            )
        else:
            raise ValueError(
                f"Unsupported EMBEDDER_BACKEND={EMBEDDER_BACKEND}. "
                "Use 'simcse' or 'sentence-transformers'."
            )
    return _embed_model


def _embed_model_source() -> str:
    return EMBED_MODEL_PATH if os.path.exists(EMBED_MODEL_PATH) else BASE_MODEL_EMBED


def _index_config_signature() -> str:
    payload = {
        "embedding_model": _embed_model_source(),
        "embedder_backend": EMBEDDER_BACKEND,
        "embedding_pipeline": EMBEDDING_PIPELINE_VERSION,
        "text_processing": TEXT_PROCESSING_VERSION,
        "simcse_pooling": SIMCSE_POOLING,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_summary_pipeline():
    global _summary_pipeline
    if _summary_pipeline is None and os.path.exists(SUMMARY_MODEL_PATH):
        from transformers import pipeline
        _summary_pipeline = pipeline(
            "summarization", model=SUMMARY_MODEL_PATH,
            tokenizer=SUMMARY_MODEL_PATH, max_new_tokens=128, device=-1,
        )
    return _summary_pipeline


def get_classifier():
    global _classifier, _label_map
    if _classifier is None and os.path.exists(CLASSIFY_MODEL_PATH):
        from transformers import pipeline
        _classifier = pipeline(
            "text-classification", model=CLASSIFY_MODEL_PATH,
            tokenizer=CLASSIFY_MODEL_PATH, device=-1,
        )
        label_map_path = os.path.join(CLASSIFY_MODEL_PATH, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path) as f:
                _label_map = json.load(f)
    return _classifier, _label_map


def _metadata_matches(meta: dict | None, where: dict | None) -> bool:
    if not where:
        return True
    if not meta:
        return False
    for key, value in where.items():
        actual = meta.get(key)
        if isinstance(value, dict) and "$in" in value:
            if actual not in value["$in"]:
                return False
        elif actual != value:
            return False
    return True


class PineconeCollectionAdapter:
    """Small collection facade over Pinecone plus a local chunk cache for BM25."""

    FETCH_BATCH_SIZE = 50
    UPSERT_BATCH_SIZE = 50
    MAX_UPSERT_RETRIES = 3
    MAX_FETCH_RETRIES = 3
    DELETE_BATCH_SIZE = 1000

    def __init__(self, index, namespace: str, cache_path: str) -> None:
        self.index = index
        self.namespace = namespace
        self.cache_path = cache_path
        self._cache: dict[str, dict] | None = None

    def _load_cache(self) -> dict[str, dict]:
        if self._cache is not None:
            return self._cache
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            raw = {}
        self._cache = raw if isinstance(raw, dict) else {}
        return self._cache

    def _save_cache(self) -> None:
        if self._cache is None:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp_path = f"{self.cache_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.cache_path)

    @staticmethod
    def _clean_metadata(meta: dict | None) -> dict:
        return {k: v for k, v in (meta or {}).items() if v is not None}

    def _upsert_with_retry(self, vectors: list[dict]) -> None:
        for attempt in range(1, self.MAX_UPSERT_RETRIES + 1):
            try:
                self.index.upsert(vectors=vectors, namespace=self.namespace)
                return
            except Exception as exc:
                if attempt >= self.MAX_UPSERT_RETRIES:
                    raise
                wait_seconds = 2 ** (attempt - 1)
                logger.warning(
                    "Pinecone upsert failed (%s/%s): %s; retrying in %ss",
                    attempt,
                    self.MAX_UPSERT_RETRIES,
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

    def _fetch_with_retry(self, ids: list[str]):
        for attempt in range(1, self.MAX_FETCH_RETRIES + 1):
            try:
                return self.index.fetch(ids=ids, namespace=self.namespace)
            except Exception as exc:
                if attempt >= self.MAX_FETCH_RETRIES:
                    raise
                wait_seconds = 2 ** (attempt - 1)
                logger.warning(
                    "Pinecone fetch failed (%s/%s): %s; retrying in %ss",
                    attempt,
                    self.MAX_FETCH_RETRIES,
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

    def count(self) -> int:
        try:
            stats = self.index.describe_index_stats()
            namespaces = getattr(stats, "namespaces", None)
            if namespaces is None and isinstance(stats, dict):
                namespaces = stats.get("namespaces", {})
            ns_stats = namespaces.get(self.namespace) if namespaces else None
            if ns_stats is None:
                return 0
            if isinstance(ns_stats, dict):
                return int(ns_stats.get("vector_count", 0))
            return int(getattr(ns_stats, "vector_count", 0))
        except Exception:
            return len(self._load_cache())

    def add(self, ids: list[str], embeddings: list, documents: list[str], metadatas: list[dict]) -> None:
        vectors = []
        cache = self._load_cache()
        for id_, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            meta = self._clean_metadata(metadata)
            vectors.append({"id": id_, "values": embedding, "metadata": {**meta, "document": document}})
            cache[id_] = {"document": document, "metadata": meta}
        if vectors:
            for start in range(0, len(vectors), self.UPSERT_BATCH_SIZE):
                self._upsert_with_retry(vectors[start : start + self.UPSERT_BATCH_SIZE])
            self._save_cache()

    def update(self, ids: list[str], embeddings: list, documents: list[str], metadatas: list[dict]) -> None:
        self.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        for start in range(0, len(ids), self.DELETE_BATCH_SIZE):
            try:
                self.index.delete(ids=ids[start : start + self.DELETE_BATCH_SIZE], namespace=self.namespace)
            except Exception as exc:
                if "Namespace not found" not in str(exc):
                    raise
                logger.info("Pinecone namespace '%s' does not exist yet; skipping stale delete.", self.namespace)
                break
        cache = self._load_cache()
        changed = False
        for id_ in ids:
            changed = cache.pop(id_, None) is not None or changed
        if changed:
            self._save_cache()

    def get(
        self,
        ids: list[str] | None = None,
        include: list[str] | None = None,
        where: dict | None = None,
        limit: int | None = None,
    ) -> dict:
        include = include or []
        cache = self._load_cache()
        result_ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        if ids is not None:
            vectors = {}
            for start in range(0, len(ids), self.FETCH_BATCH_SIZE):
                batch_ids = ids[start : start + self.FETCH_BATCH_SIZE]
                fetch_response = self._fetch_with_retry(batch_ids) if batch_ids else None
                batch_vectors = getattr(fetch_response, "vectors", None) if fetch_response is not None else {}
                if batch_vectors is None and isinstance(fetch_response, dict):
                    batch_vectors = fetch_response.get("vectors", {})
                vectors.update(batch_vectors or {})
            for id_ in ids:
                vector = vectors.get(id_) if vectors else None
                cached = cache.get(id_)
                metadata = getattr(vector, "metadata", None) if vector is not None else None
                if metadata is None and isinstance(vector, dict):
                    metadata = vector.get("metadata")
                if vector is None and cached is None:
                    continue
                document = (metadata or {}).get("document") if metadata else None
                if document is None and cached:
                    document = cached.get("document", "")
                meta = self._clean_metadata(metadata or (cached or {}).get("metadata", {}))
                meta.pop("document", None)
                if not _metadata_matches(meta, where):
                    continue
                result_ids.append(id_)
                if "documents" in include:
                    documents.append(document or "")
                if "metadatas" in include:
                    metadatas.append(meta)
                if limit and len(result_ids) >= limit:
                    break
        else:
            for id_, row in cache.items():
                meta = self._clean_metadata(row.get("metadata", {}))
                if not _metadata_matches(meta, where):
                    continue
                result_ids.append(id_)
                if "documents" in include:
                    documents.append(row.get("document", ""))
                if "metadatas" in include:
                    metadatas.append(meta)
                if limit and len(result_ids) >= limit:
                    break

        response = {"ids": result_ids}
        if "documents" in include:
            response["documents"] = documents
        if "metadatas" in include:
            response["metadatas"] = metadatas
        return response

    def query(
        self,
        query_embeddings: list,
        n_results: int,
        include: list[str] | None = None,
        where: dict | None = None,
    ) -> dict:
        response = self.index.query(
            vector=query_embeddings[0],
            top_k=n_results,
            namespace=self.namespace,
            filter=where,
            include_metadata=True,
        )
        matches = getattr(response, "matches", None)
        if matches is None and isinstance(response, dict):
            matches = response.get("matches", [])

        ids: list[str] = []
        metadatas: list[dict] = []
        distances: list[float] = []
        documents: list[str] = []
        for match in matches or []:
            match_id = getattr(match, "id", None) if not isinstance(match, dict) else match.get("id")
            score = getattr(match, "score", None) if not isinstance(match, dict) else match.get("score")
            metadata = getattr(match, "metadata", None) if not isinstance(match, dict) else match.get("metadata")
            meta = self._clean_metadata(metadata)
            document = meta.pop("document", "")
            ids.append(match_id)
            metadatas.append(meta)
            distances.append(1 - float(score or 0))
            documents.append(document)

        result = {"ids": [ids]}
        include = include or []
        if "metadatas" in include:
            result["metadatas"] = [metadatas]
        if "distances" in include:
            result["distances"] = [distances]
        if "documents" in include:
            result["documents"] = [documents]
        return result


def _get_pinecone_collection() -> PineconeCollectionAdapter:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is required when VECTOR_DB=pinecone.")
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            deletion_protection="disabled",
        )
    for _ in range(60):
        desc = pc.describe_index(PINECONE_INDEX_NAME)
        status = getattr(desc, "status", None)
        if status is None and isinstance(desc, dict):
            status = desc.get("status", {})
        ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
        if ready:
            dimension = desc.get("dimension") if isinstance(desc, dict) else getattr(desc, "dimension", None)
            if dimension and int(dimension) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Pinecone index dimension mismatch: {dimension} != EMBEDDING_DIM={EMBEDDING_DIM}."
                )
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"Pinecone index is not ready: {PINECONE_INDEX_NAME}")
    return PineconeCollectionAdapter(
        index=pc.Index(PINECONE_INDEX_NAME),
        namespace=PINECONE_NAMESPACE,
        cache_path=PINECONE_CACHE_PATH,
    )


def get_vector_collection():
    global _vector_collection
    if _vector_collection is None:
        if VECTOR_DB != "pinecone":
            raise RuntimeError(
                f"Pinecone is required for the production vector store; got VECTOR_DB={VECTOR_DB!r}."
            )
        _vector_collection = _get_pinecone_collection()
    return _vector_collection


def classify_notice(title: str, body: str) -> str:
    clf, label_map = get_classifier()
    if clf is None:
        return infer_category(title, body)
    try:
        result   = clf(f"{title} {body[:200]}", truncation=True)[0]
        label_id = result["label"].replace("LABEL_", "")
        return label_map.get(label_id, "기타")
    except Exception:
        return infer_category(title, body)


def load_notices_cache() -> list[dict]:
    if os.path.exists(NOTICES_CACHE_PATH):
        with open(NOTICES_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def _load_index_manifest() -> dict:
    if not os.path.exists(INDEX_MANIFEST_PATH):
        return {"version": 1, "revision": 0, "index_config": {}, "notices": {}}
    try:
        with open(INDEX_MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("인덱스 manifest를 읽지 못해 전체 갱신 대상으로 처리합니다.")
        return {"version": 1, "revision": 0, "index_config": {}, "notices": {}}
    manifest.setdefault("version", 1)
    manifest.setdefault("revision", 0)
    manifest.setdefault("index_config", {})
    manifest.setdefault("notices", {})
    return manifest


def _save_index_manifest(manifest: dict) -> None:
    os.makedirs(os.path.dirname(INDEX_MANIFEST_PATH), exist_ok=True)
    manifest["revision"] = int(manifest.get("revision", 0)) + 1
    tmp_path = f"{INDEX_MANIFEST_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, INDEX_MANIFEST_PATH)


def get_index_fingerprint() -> tuple[int, int, str]:
    """Return a cheap fingerprint for search-side cache invalidation."""
    manifest = _load_index_manifest()
    return (
        get_vector_collection().count(),
        int(manifest.get("revision", 0)),
        manifest.get("index_config", {}).get("signature", ""),
    )


def _notice_doc_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _notice_content_hash(item: dict) -> str:
    payload = {
        "url": item.get("url", ""),
        "title": item.get("title", ""),
        "body": item.get("body", ""),
        "date": item.get("date", ""),
        "category": item.get("category", ""),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def index_notices(
    notices: list[dict],
    force: bool = False,
    sync_deletions: bool = False,
    notice_batch_size: int = 20,
    embed_batch_size: int = 16,
) -> int:
    """Index notices into the configured vector DB. Returns count of newly indexed notices."""
    collection = get_vector_collection()
    manifest   = _load_index_manifest()
    manifest_notices = manifest["notices"]
    notice_batch_size = max(1, notice_batch_size)
    embed_batch_size = max(1, embed_batch_size)
    index_config = {
        "signature": _index_config_signature(),
        "embedding_model": _embed_model_source(),
        "embedder_backend": EMBEDDER_BACKEND,
        "embedding_pipeline": EMBEDDING_PIPELINE_VERSION,
        "text_processing": TEXT_PROCESSING_VERSION,
        "simcse_pooling": SIMCSE_POOLING,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    config_changed = manifest.get("index_config", {}).get("signature") != index_config["signature"]
    if config_changed:
        logger.info("인덱스 설정이 변경되어 전체 재색인 대상으로 처리합니다.")
        manifest["index_config"] = index_config

    # 1. 전체 doc_id/content_hash 계산
    all_records = [
        (_notice_doc_id(item["url"]), _notice_content_hash(item), item)
        for item in notices
    ]
    current_doc_ids = {doc_id for doc_id, _, _ in all_records}
    current_urls = {item["url"] for _, _, item in all_records}
    existing_ids_by_url: dict[str, list[str]] = {}

    stale_chunk_ids: list[str] = []
    if sync_deletions:
        existing = collection.get(include=["metadatas"])
        for existing_id, meta in zip(existing["ids"], existing["metadatas"]):
            if meta and meta.get("url"):
                existing_ids_by_url.setdefault(meta["url"], []).append(existing_id)
        for url, ids in existing_ids_by_url.items():
            if url not in current_urls:
                stale_chunk_ids.extend(ids)

        stale_doc_ids = set(manifest_notices) - current_doc_ids
        for stale_doc_id in stale_doc_ids:
            entry = manifest_notices.get(stale_doc_id, {})
            stale_chunk_ids.extend(entry.get("chunk_ids", []))
            stale_chunk_ids.append(stale_doc_id)
        if stale_chunk_ids:
            existing_stale = set(collection.get(ids=list(set(stale_chunk_ids)))["ids"])
            if existing_stale:
                collection.delete(ids=list(existing_stale))
        for stale_doc_id in stale_doc_ids:
            manifest_notices.pop(stale_doc_id, None)
        if stale_doc_ids:
            logger.info("소스에서 사라진 공지 %d건을 인덱스 manifest에서 제거했습니다.", len(stale_doc_ids))

    # 2. 존재 확인: N번 개별 쿼리 → 1번 배치 쿼리
    chunk0_ids      = [f"{doc_id}_0" for doc_id, _, _ in all_records]
    existing_chunk0 = set(collection.get(ids=chunk0_ids)["ids"]) if chunk0_ids else set()

    pending: list[tuple[str, str, dict]] = []
    for doc_id, content_hash, item in all_records:
        indexed = f"{doc_id}_0" in existing_chunk0
        manifest_entry = manifest_notices.get(doc_id)
        unchanged = (
            manifest_entry
            and manifest_entry.get("content_hash") == content_hash
            and manifest_entry.get("index_config_signature") == index_config["signature"]
            and indexed
        )
        if force or config_changed or not unchanged:
            pending.append((doc_id, content_hash, item))

    # 3. 구버전 포맷(청크 없는 doc_id) 일괄 삭제 — 1번 쿼리
    if pending:
        old_ids      = [doc_id for doc_id, _, _ in pending]
        existing_old = set(collection.get(ids=old_ids)["ids"])
        to_delete    = [did for did in old_ids if did in existing_old]
        if to_delete:
            collection.delete(ids=to_delete)

    if not pending:
        if sync_deletions and stale_chunk_ids:
            _save_index_manifest(manifest)
        return 0

    if not existing_ids_by_url and any(
        not manifest_notices.get(doc_id, {}).get("chunk_ids")
        for doc_id, _, _ in pending
    ):
        existing = collection.get(include=["metadatas"])
        for existing_id, meta in zip(existing["ids"], existing["metadatas"]):
            if meta and meta.get("url"):
                existing_ids_by_url.setdefault(meta["url"], []).append(existing_id)

    # pending이 있을 때만 무거운 임베딩 모델을 로드한다.
    model = get_embed_model()
    logger.info(
        "임베딩 시작! 총 %d개 공지 처리 예정 (notice_batch=%d, embed_batch=%d)",
        len(pending),
        notice_batch_size,
        embed_batch_size,
    )

    total_indexed = 0

    for batch_start in range(0, len(pending), notice_batch_size):
        batch = pending[batch_start : batch_start + notice_batch_size]
        batch_no = batch_start // notice_batch_size + 1
        total_batches = (len(pending) + notice_batch_size - 1) // notice_batch_size

        notice_chunks: list[tuple[str, str, list[str], list[str], list[dict]]] = []
        batch_delete_ids: list[str] = []
        docs: list[str] = []

        for doc_id, content_hash, item in batch:
            entry = manifest_notices.get(doc_id, {})
            ids_to_delete = entry.get("chunk_ids", [])
            if not ids_to_delete:
                ids_to_delete = existing_ids_by_url.get(item["url"], [])
            if ids_to_delete:
                batch_delete_ids.extend(ids_to_delete)

            body     = item.get("body", "")
            inferred_category = classify_notice(item["title"], body)
            existing_category = item.get("category")
            category = (
                inferred_category
                if inferred_category == "봉사/서포터즈" and existing_category in {None, "", "국제교류", "기타"}
                else existing_category or inferred_category
            )
            chunks   = chunk_text(f"제목: {item['title']}\n\n{body}")
            meta     = {"title": item["title"], "url": item["url"],
                        "date": item["date"], "category": category}
            chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            notice_chunks.append(
                (doc_id, content_hash, chunk_ids, chunks, [meta] * len(chunks))
            )
            docs.extend(chunks)

        if batch_delete_ids:
            collection.delete(ids=list(dict.fromkeys(batch_delete_ids)))

        logger.info(
            "공지 배치 인코딩 시작: %d/%d 배치 (%d개 공지, %d개 청크, device=%s, embed_batch=%d)",
            batch_no,
            total_batches,
            len(batch),
            len(docs),
            getattr(model, "_device", getattr(model, "device", "unknown")),
            embed_batch_size,
        )
        embeddings = model.encode(
            docs,
            batch_size=embed_batch_size,
            show_progress_bar=True,
        ).tolist()

        ids: list[str] = []
        metas: list[dict] = []
        for _, _, n_ids, _, n_metas in notice_chunks:
            ids.extend(n_ids)
            metas.extend(n_metas)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=docs,
            metadatas=metas,
        )
        total_indexed += len(batch)
        for doc_id, content_hash, n_ids, _, _ in notice_chunks:
            manifest_notices[doc_id] = {
                "content_hash": content_hash,
                "index_config_signature": index_config["signature"],
                "chunk_ids": n_ids,
            }
        _save_index_manifest(manifest)
        del embeddings, ids, metas, docs, notice_chunks
        _release_torch_cache()
        logger.info(
            "공지 배치 저장 완료: %d/%d 배치, 누적 %d/%d개 공지",
            batch_no,
            total_batches,
            total_indexed,
            len(pending),
        )

    return total_indexed
