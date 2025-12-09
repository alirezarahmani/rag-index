import time
from typing import Tuple
import numpy as np
import faiss
from tqdm import tqdm

def load_or_generate_embeddings(
    n_vectors: int = 200_000,
    dim: int = 384,
    n_queries: int = 200,
    seed: int = 42,
    path: str = "data/embeddings.npy",
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    try:
        xb = np.load(path)
        assert xb.ndim == 2
        assert xb.shape[1] == dim
        print(f"[load] Loaded embeddings from {path}: {xb.shape}")
    except Exception:
        print(f"[gen] Generating random embeddings: N={n_vectors}, d={dim}")
        xb = rng.normal(size=(n_vectors, dim)).astype("float32")
        os.makedirs("data", exist_ok=True)
        np.save(path, xb)

    idx = rng.choice(xb.shape[0], size=n_queries, replace=False)
    xq = xb[idx] + 0.01 * rng.normal(size=(n_queries, dim)).astype("float32")

    return xb.astype("float32"), xq.astype("float32")

def build_flat_index(xb: np.ndarray):
    d = xb.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    return index

def build_ivf_index(xb: np.ndarray, nlist: int):
    d = xb.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    print(f"[ivf] Training IVF index with nlist={nlist}...")
    index.train(xb)
    index.add(xb)
    return index

def build_hnsw_index(xb: np.ndarray, m: int = 32, ef_construction: int = 200):
    d = xb.shape[1]
    index = faiss.IndexHNSWFlat(d, m)
    index.hnsw.efConstruction = ef_construction
    print(f"[hnsw] Building HNSW index M={m}, efConstruction={ef_construction}...")
    index.add(xb)
    return index

def recall_at_k(gt_ids, test_ids, k: int) -> float:
    nq = gt_ids.shape[0]
    hits = 0
    for i in range(nq):
        hits += len(set(gt_ids[i, :k]) & set(test_ids[i]))
    return hits / (nq * k)

def time_search(index, xq, k: int, **kwargs):
    for key, value in kwargs.items():
        if hasattr(index, key):
            setattr(index, key, value)
        elif key == "nprobe":
            faiss.ParameterSpace().set_index_parameter(index, key, value)

    start = time.perf_counter()
    D, I = index.search(xq, k)
    end = time.perf_counter()
    latency = (end - start) * 1000.0 / xq.shape[0]
    return I, latency

def benchmark_indices():
    dim = 384
    n_vectors = 200_000
    n_queries = 200
    k = 10

    ivf_nlist = 1024
    ivf_nprobe_values = [1, 4, 8, 16, 32, 64]

    hnsw_m = 32
    hnsw_efc = 200
    hnsw_ef_search_values = [16, 32, 64, 128, 256]

    xb, xq = load_or_generate_embeddings(n_vectors, dim, n_queries)
    print(f"[data] corpus={xb.shape}, queries={xq.shape}")

    print("\n=== Flat (ground truth) ===")
    flat = build_flat_index(xb)
    gt_ids, flat_lat = time_search(flat, xq, k)
    print(f"[flat] latency ≈ {flat_lat:.2f} ms/query")

    print("\n=== IVF-Flat Benchmark ===")
    ivf = build_ivf_index(xb, ivf_nlist)
    for nprobe in ivf_nprobe_values:
        faiss.ParameterSpace().set_index_parameter(ivf, "nprobe", nprobe)
        ids, lat = time_search(ivf, xq, k)
        r = recall_at_k(gt_ids, ids, k)
        print(f"[ivf] nprobe={nprobe:<3d} | recall@{k}={r:.3f} | latency={lat:.2f} ms/query")

    print("\n=== HNSW Benchmark ===")
    hnsw = build_hnsw_index(xb, hnsw_m, hnsw_efc)
    for ef_s in hnsw_ef_search_values:
        hnsw.hnsw.efSearch = ef_s
        ids, lat = time_search(hnsw, xq, k)
        r = recall_at_k(gt_ids, ids, k)
        print(f"[hnsw] efSearch={ef_s:<3d} | recall@{k}={r:.3f} | latency={lat:.2f} ms/query")

    print("\nDone.")

if __name__ == "__main__":
    benchmark_indices()
