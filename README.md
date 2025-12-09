# **RAG Index Benchmark Suite**

**Stop Guessing. Measure Your Retrieval Layer.**

Most RAG systems don't fail because of embeddings.
They fail because the **index** silently drops the documents that matter.

This repository contains a **fully reproducible benchmark** that shows exactly how Flat, IVF, and HNSW behave across different dataset sizes, different query loads, and different parameter sweeps. If you care about **recall**, **latency**, or **scaling real systems**, this suite gives you the numbers, not the vibes.

> 🔗 **Full article:**
> **“RAG Isn't About Embeddings — It's About the Index You Chose”**
> [https://nidly.substack.com/p/rag-isnt-about-embeddings](https://nidly.substack.com/p/rag-isnt-about-embeddings)

---

## 🚀 **What This Repo Gives You**

This benchmark suite is built to answer one question:

**“Which index should I actually use for *my* dataset and *my* latency budget?”**

Inside the suite:

### ✔ **[FAISS](https://nidly.substack.com/p/what-is-faiss-building-your-own-zillow?r=a3p8i) Benchmarking Engine**

* Flat (Exact)
* IVF (tunable `nlist`, `nprobe`)
* HNSW (`M`, `efConstruction`, `efSearch`)

### ✔ **True Recall Measurement**

We compute ground truth using Flat search, then measure how IVF and HNSW degrade relative to perfection.

### ✔ **Latency Profiling**

We record:

* p50
* p95
* p99
* throughput impact under concurrent load

### ✔ **Parameter Sweeps**

One file, already configured:

* IVF → `nprobe` sweep across 1 → 100
* HNSW → `efSearch` sweep across 10 → 300

You get the **Pareto Frontier**:
**Recall vs Latency**, the only curve that actually matters in Retrieval.

### ✔ **Production-Ready Code**

Fully self-contained, reproducible, and clean enough to drop into your own experiments or articles.

---

## 🧠 **Why This Repo Exists**

Because every team that builds RAG eventually hits this moment:

> “The LLM is fine. The embeddings are fine.
> So why the hell are we getting incomplete answers?”

This repo is the answer:
because **your index configuration is silently sabotaging you**.

If you don't measure:

* the **true recall gap**
* the **latency ceiling**
* the **cluster imbalance**
* the **graph search depth trade-offs**

…you're building blind.

This suite shows you the real behavior of your retrieval layer, not the marketing slides.

---

## 📊 **Example Insights You'll Discover**

With this benchmark, you'll see things like:

* IVF with `nprobe=1` can drop recall into the 40–60% range.
* HNSW becomes unreliable at low `efSearch`, even though it “looks fast”.
* Flat is unbeatable under 300k vectors but collapses past 1M.
* Increasing `nlist` without rebalancing clusters kills IVF accuracy.
* Latency curves behave differently under concurrent load vs single-shot queries.

None of these show up in blog posts.
All of them show up in real production systems.

---

## 🛠 **How to Run**

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the benchmark

```bash
python bench_faiss_indices.py
```

Results are saved as JSON + console output.

### 3. Plot or analyze

You can plug outputs directly into your notebook or monitoring tools.

---

## 🏆 **Who This Repo Is For**

* **Search engineers** shipping RAG into production
* **AI engineers** who want numbers, not vibes
* **CTOs & architects** modeling system latency budgets
* **Researchers** studying ANN behavior under real constraints
* **Anyone** who's tired of guessing which index to use

If you care about correctness, performance, or scale, this repo saves you weeks.

---

## 📬 **If You Want the Deep Dive**

Read the full article series about retrieval engineering, index tuning, and real-world RAG failures:

👉 **[https://nidly.substack.com](https://nidly.substack.com)**

Here's the star of the show for this repo:
**“RAG Isn't About Embeddings — It's About the Index You Chose”**
A breakdown of *why* embeddings get too much credit and indexes get too little.

---

## ❤️ **Support the Work**

If this repo helps you:

⭐ Star the repo
📝 Subscribe to the Substack
🔗 Share the articles with your team

It keeps the research flowing and the benchmarks improving.

---

## 🧩 **Future Plans**

* GPU acceleration benchmarking
* PQ & OPQ compression benchmarks
* Milvus & Weaviate benchmark adapters
* Hybrid retrieval (BM25 + dense vectors)
* Cross-encoder re-ranking scoring suite

Stay tuned. This is just version zero.

---
