import pytest

from ops_copilot.chunk import chunk_docs
from ops_copilot.index import Index, corpus_hash
from ops_copilot.ingest import load_docs
from ops_copilot.retrieve import bm25_search, dense_search, retrieve, rrf_merge

from .conftest import FIXTURE_DOCS


@pytest.fixture
def index(tmp_path, embeddings):
    return Index(chunk_docs(load_docs(FIXTURE_DOCS)), embeddings, tmp_path / "chroma", "fake")


def test_rrf_merge_hand_computed():
    # lists: [a, b] and [b, c], k=60
    # a = 1/61 ; b = 1/62 + 1/61 ; c = 1/62
    out = rrf_merge([["a", "b"], ["b", "c"]], k=60)
    assert out["a"] == pytest.approx(1 / 61)
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["c"] == pytest.approx(1 / 62)
    assert out["a"] == pytest.approx(0.016393442, abs=1e-9)
    assert out["b"] == pytest.approx(0.032522474, abs=1e-9)
    assert sorted(out, key=lambda x: -out[x]) == ["b", "a", "c"]


def test_rrf_merge_k_parameter():
    assert rrf_merge([["a"]], k=10)["a"] == pytest.approx(1 / 11)


def test_bm25_and_dense_find_the_right_section(index, embeddings):
    q = "emergency purchase orders ratified"
    assert bm25_search(index, q, 3)[0][0] == "PR-001#2.2#0"
    top = dense_search(index, embeddings.embed_query(q), 3)
    assert top[0][0] == "PR-001#2.2#0"
    assert 0.0 < top[0][1] <= 1.0 + 1e-6


def test_dense_score_is_cosine_similarity(index, embeddings):
    c = index.chunks[0]
    top = dense_search(index, embeddings.embed_query(c.text), 1)[0]
    assert top[0] == c.id
    assert top[1] == pytest.approx(1.0, abs=1e-4)


def test_bm25_excludes_zero_score_chunks(index):
    assert bm25_search(index, "zzzunknownterm", 5) == []


def test_retrieve_hybrid_rank_fields(index):
    r = retrieve(index, "damaged cartons quarantined bay", "hybrid", top_k=4, rrf_k=60)
    top = r.hits[0]
    assert top.chunk.id == "WH-003#2.2#0"
    assert top.bm25_rank == 1 and top.dense_rank == 1
    assert top.rrf == pytest.approx(2 / 61)
    rrfs = [h.rrf for h in r.hits]
    assert rrfs == sorted(rrfs, reverse=True)
    assert len(r.hits) <= 4
    assert r.confidence == pytest.approx(top.dense_score)
    assert set(r.timings_ms) == {"embed", "bm25", "dense", "rrf"}
    assert all(v >= 0 for v in r.timings_ms.values())


def test_retrieve_modes_order_and_candidate_cutoff(index):
    q = "damaged cartons quarantined bay"
    bm = retrieve(index, "the and in orders", "bm25", top_k=3)
    assert [h.bm25_rank for h in bm.hits] == [1, 2, 3]
    de = retrieve(index, q, "dense", top_k=3)
    assert [h.dense_rank for h in de.hits] == [1, 2, 3]
    scores = [h.dense_score for h in de.hits]
    assert scores == sorted(scores, reverse=True)
    # With 1 candidate per retriever, anything outside rank 1 has rank None.
    one = retrieve(index, q, "hybrid", top_k=5, candidates=1)
    assert sum(h.bm25_rank is not None for h in one.hits) == 1
    assert sum(h.dense_rank is not None for h in one.hits) == 1


def test_retrieve_rejects_unknown_mode(index):
    with pytest.raises(ValueError):
        retrieve(index, "x", "magic")


def test_chroma_persists_in_tmp_dir_and_rebuilds_only_on_change(tmp_path, embeddings):
    chunks = chunk_docs(load_docs(FIXTURE_DOCS))
    d = tmp_path / "store"
    first = Index(chunks, embeddings, d, "m1")
    assert first.rebuilt and (d / "chroma.sqlite3").exists()
    assert not Index(chunks, embeddings, d, "m1").rebuilt  # same content: reused
    assert Index(chunks, embeddings, d, "m2").rebuilt  # embed model changed
    assert Index(chunks[:-1], embeddings, d, "m2").rebuilt  # corpus changed
    assert corpus_hash(chunks, "m") != corpus_hash(chunks[:-1], "m")
