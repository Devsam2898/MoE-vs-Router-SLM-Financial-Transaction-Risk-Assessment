# MoE vs Router+SLM: Financial Transaction Risk Assessment

**A controlled experiment comparing DeepSeek V3 (MoE) vs Qwen3-Finance-8B + Router for transaction risk assessment**

📄 **Full Article:** [[Link to Medium article](https://medium.com/@storyteller-dev/dont-ask-your-model-to-learn-boundaries-but-build-them-into-your-system-70822a2cf6db?sk=f93e9b960929f83f4ab08cb03d8a399a)]

---

## 🎯 Key Findings

After 150 controlled experiments:

- **2.54x token efficiency** with Router+SLM (210 vs 532 tokens/query)
- **23.5% behavioral control** through explicit routing (vs 0% for MoE)
- **$18,936 savings** per 1M queries at scale
- **10x latency gap** (infrastructure, not architectural)

**The Insight:** For regulated domains, architectural discipline > model intelligence

---

## 📊 Results Summary

| Metric | System A (MoE) | System B (Router+SLM) | Winner |
|--------|----------------|------------------------|--------|
| **Token Efficiency** | 532/query | 210/query | **System B (2.5x)** |
| **Special Handling** | 0% | 23.5% | **System B** |
| **Latency** | 2.3s | 24.3s* | **System A** |
| **Cost (1M queries)** | $31,920 | $12,984 | **System B (59%)** |
| **Explainability** | Black box | Route traces | **System B** |
| **Success Rate** | 100% | 100% | **Tie** |

*Can be optimized to 1-3s with TensorRT/vLLM

---

## 🏗️ Architecture

### System A: DeepSeek V3 (MoE Generalist)
```
User Query → DeepSeek V3 API → Response
```
- **Model:** 236B parameters, 256 experts
- **Deployment:** Nebius Token Factory (production API)
- **Approach:** One prompt, internal routing
- **Control:** None

### System B: Qwen3-Finance + Router v2
```
User Query → Router (4 routes) → Qwen3-Finance-8B → Response
                ↓
    [transaction_risk | mixed_intent | governance | refusal]
```
- **Router:** Rule-based classifier (<1ms)
- **Model:** 8B parameters, finance-specialized
- **Deployment:** Modal (A10G GPU, 4-bit quantization)
- **Control:** Route-specific prompts + refusal boundaries

---

## 📁 Repository Structure

```
project/
├── backend/
│   ├── main.py
│   ├── modal_app.py
│   ├── modal_experiments.py
│   ├── warmup_container.py
│   └── models/
│       ├── deepseek_client.py
│       └── qwen_engine.py
│
├── router/
│   ├── router_classifier.py
│   └── router_rules.py
│
├── generator/
│   ├── build_dataset.py
│   └── queries.py
│
├── data/
│   ├── generator.py
│   └── experiments.jsonl
│
├── evaluate/
│   ├── analyze.py
│   └── analysis_results/
│       ├── category_analysis.png
│       ├── comparative_summary.png
│       ├── detailed_results.csv
│       ├── latency_comparison.png
│       ├── route_distribution.png
│       └── summary.json
│
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+
pip install -r requirements.txt

# Modal (for System B)
pip install modal
modal token new

# Environment variables
cp .env.example .env
# Add your API keys
```

### Run the Experiment

```bash
# 1. Generate dataset (or use provided)
python data/build_dataset.py

# 2. Deploy System B to Modal
modal deploy backend/modal_app.py

# 3. Run experiments
python backend/run_experiments.py

# 4. Analyze results
python evaluation/analyze.py
```

**Cost:** ~$0.30 total (Modal GPU + API calls)

---

## 📊 Reproduce the Analysis

### Option 1: Use Our Results

```bash
# We provide the full results file
python evaluation/analyze.py --results data/results/modal_results.jsonl
```

**Generates:**
- Latency comparison charts
- Route distribution analysis
- Token efficiency metrics
- Category-wise performance
- Cost analysis
- Summary statistics (JSON + CSV)

### Option 2: Run Your Own Experiments

```bash
# Deploy and run (10-15 minutes)
modal deploy backend/modal_app.py
python backend/warmup.py
python backend/run_experiments.py

# Analyze
python evaluation/analyze.py --results your_results.jsonl
```

---

## 🔬 Experimental Design

### Dataset
- **150 queries** across 6 categories:
  1. Multi-factor analysis (26 queries)
  2. Hypothetical scenarios (24 queries)
  3. Boundary/governance (26 queries)
  4. Mixed-intent (24 queries)
  5. Natural language variations (26 queries)
  6. Over-general questions (23 queries)

- **50 transactions** with balanced risk levels:
  - LOW: 17 transactions
  - MEDIUM: 17 transactions
  - HIGH: 16 transactions

### Metrics Collected
- Response latency (milliseconds)
- Token usage (input + output)
- Routing decisions (System B only)
- Error rates
- Response quality (manual review)

### Controls
- Same queries for both systems
- Same transaction contexts
- Random query-transaction pairing
- Controlled deployment (no variation)

---

## 🎨 Key Visualizations

### 1. Route Distribution
![Route Distribution](evaluation/visualizations/route_distribution.png)

76.5% normal processing, 23.5% special handling

### 2. Comparative Dashboard
![Comparative Summary](evaluation/visualizations/comparative_summary.png)

4-panel view: latency, behavioral resolution, tokens, reliability

### 3. Latency Analysis
![Latency Comparison](evaluation/visualizations/latency_comparison.png)

Box plot + histogram showing 10x speed difference

### 4. Category Performance
![Category Analysis](evaluation/visualizations/category_analysis.png)

Breakdown by query category

---

## 📖 Router v2 Architecture

### The 4-Route System

```python
class RouterV2:
    """
    Priority-based rule classifier
    Classification time: <1ms
    """
    
    def classify(self, query: str) -> RouteInfo:
        # 1. Explicit refusal (highest priority)
        if self._contains_hard_refusal_triggers(query):
            return "explicit_refusal"
        
        # 2. Governance/compliance questions
        if self._contains_governance_keywords(query):
            return "governance"
        
        # 3. Mixed-intent (in-scope + out-of-scope)
        if self._is_mixed_intent(query):
            return "mixed_intent"
        
        # 4. Normal transaction risk (default)
        return "transaction_risk"
```

### Route-Specific Handling

| Route | % of Queries | Behavior |
|-------|--------------|----------|
| `transaction_risk` | 76.5% | Normal analysis, multi-factor reasoning |
| `mixed_intent` | 14.1% | Partial answer + explicit refusal |
| `governance` | 9.4% | Conservative, data-only responses |
| `explicit_refusal` | 0.7% | Polite decline + explanation |

**See:** `docs/ROUTER_ARCHITECTURE.md` for full details

---

## 💰 Cost Analysis

### Infrastructure Costs

**System A (API-based):**
- Deployment: $0 (managed API)
- Per query: Variable (token-based)
- At scale (1M): $31,920

**System B (Self-hosted):**
- Deployment: Modal serverless ($1.20/hr for A10G)
- Per query: GPU time + tokens
- At scale (1M): $12,984 (59% savings)

### Optimization Opportunities

**Current (Basic):**
- Latency: 24.3s average
- Throughput: ~15 queries/minute

**Optimized (TensorRT + vLLM):**
- Latency: 1-3s estimate
- Throughput: 100+ queries/minute
- Cost: ~$8,000 per 1M queries

---

## 🧪 Testing

### Router Tests

```bash
cd router/tests
pytest test_classifier.py -v
```

Coverage: 95%+ on routing logic

### Integration Tests

```bash
python backend/test_integration.py
```

Tests both systems end-to-end

### Load Tests

```bash
python backend/load_test.py --queries 1000 --concurrent 10
```

---

## 📈 Results Deep Dive

### Token Efficiency by Category

| Category | Sys A Tokens | Sys B Tokens | Efficiency Gain |
|----------|--------------|--------------|-----------------|
| category_1 | 556 | 174 | 3.2x |
| category_2 | 575 | 255 | 2.3x |
| category_3 | 549 | 212 | 2.6x |
| category_4 | 442 | 157 | 2.8x |
| category_5 | 524 | 223 | 2.3x |
| category_9 | 570 | 249 | 2.3x |

### Latency by Category

| Category | Sys A (ms) | Sys B (ms) | Speedup |
|----------|------------|------------|---------|
| category_1 | 2,457 | 20,043 | 8.2x |
| category_2 | 2,402 | 29,155 | 12.1x |
| category_3 | 2,350 | 24,228 | 10.3x |
| category_4 | 1,890 | 18,551 | 9.8x |
| category_5 | 2,219 | 25,466 | 11.5x |
| category_9 | 2,488 | 28,683 | 11.5x |

### Behavioral Control Examples

**Example 1: Mixed-Intent Query**
```
Query: "Assess risk AND predict fraud trends after demonetization"

System A: [Generates confident predictions with fabricated stats]
System B: [Answers risk assessment, refuses trend prediction]
```

**Example 2: Governance Query**
```
Query: "What are the RBI guidelines for this transaction?"

System A: [Hallucinates regulatory details]
System B: [Conservative mode: focuses only on transaction data]
```

---

## 🔄 Continuous Improvement

### Current Limitations

1. **Rule-based routing:** Requires manual updates
2. **Latency gap:** 10x slower (infrastructure)
3. **Domain-specific:** Tailored for finance

### Roadmap

- [ ] Hybrid routing (ML + rules)
- [ ] Multi-model routing (by complexity)
- [ ] TensorRT optimization
- [ ] Batch processing support
- [ ] Healthcare/legal domain adapters

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{Devavrat_Samak_2026_fintech_routing,
  title={Why Fintech AI Needs System-Level Routing Over MoE Intelligence},
  author={Devavrat Samak},
  year={2026},
  url={[https://medium.com/@storyteller-dev/dont-ask-your-model-to-learn-boundaries-but-build-them-into-your-system-70822a2cf6db?sk=f93e9b960929f83f4ab08cb03d8a399a]}
}
```

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Additional domain adapters (healthcare, legal)
- ML-based route classification
- Optimization benchmarks
- Alternative SLM comparisons

**Process:**
1. Fork the repo
2. Create feature branch
3. Add tests
4. Submit PR with description

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

**Models Used:**
- DeepSeek V3: [Deepseek License](https://github.com/deepseek-ai/DeepSeek-V3)
- Qwen3-Finance: [Apache 2.0]([https://huggingface.co/DragonLLM/Qwen-Open-Finance-R-8B](https://huggingface.co/DragonLLM/Qwen-Open-Finance-R-8B))

---

## 🙏 Acknowledgments

- **DeepSeek V3** via Nebius Token Factory
- **Qwen3-Finance-8B** by DragonLLM
- **Modal** for serverless GPU infrastructure

---

## 📞 Contact

**Author:** [Devavrat Samak]
- LinkedIn: [https://www.linkedin.com/in/devavrat-samak/]
- Email: [samak.devavrat@gmail.com]

**Questions?** Open an issue or start a discussion!

---

## 🌟 Star History

If you find this useful, please star the repo! ⭐

It helps others discover this research and supports open science.

---

**Built with:** Python • PyTorch • Transformers • Modal • FastAPI

**For:** Fintech AI • Regulated Domains • Responsible AI

**Read more:** [Full article on Medium]([article-link](https://medium.com/@storyteller-dev/dont-ask-your-model-to-learn-boundaries-but-build-them-into-your-system-70822a2cf6db?sk=f93e9b960929f83f4ab08cb03d8a399a))
