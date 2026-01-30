# Backend API - MoE vs SLM Comparison

FastAPI-based backend that orchestrates System A (DeepSeek V3 MoE) and System B (Router + Qwen Finance SLM) for comparative analysis.

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│         (main.py)                       │
└────────┬───────────────────────┬────────┘
         │                       │
    ┌────▼────────┐        ┌────▼─────────┐
    │  System A   │        │   System B   │
    │   Pipeline  │        │   Pipeline   │
    └────┬────────┘        └────┬─────────┘
         │                      │
    ┌────▼────────┐        ┌────▼─────────┐
    │  DeepSeek   │        │   Router     │
    │  V3 Client  │        └────┬─────────┘
    │  (Nebius)   │             │
    └─────────────┘        ┌────▼─────────┐
                           │ Qwen Finance │
                           │   Engine     │
                           └──────────────┘
```

## 📦 Components

### 1. **main.py** - FastAPI Application
- REST API with comparison endpoints
- Parallel execution of both systems
- Health checks and monitoring
- Batch processing support

### 2. **models/deepseek_client.py** - System A
- Wrapper for Nebius Token Factory API
- DeepSeek V3 (MoE) inference
- Generic financial analyst prompt
- No routing, no refusal logic

### 3. **models/qwen_engine.py** - System B
- Loads Qwen-Open-Finance-R-8B locally
- Route-specific system prompts
- 4-bit quantization support
- Deterministic generation (temp=0.1)

### 4. **Router Integration**
- Uses classifier from `/router` module
- 4-route system (transaction_risk, mixed_intent, governance, explicit_refusal)
- Provides handling instructions to SLM

## 🚀 Quick Start

### Prerequisites

```bash
# Set Nebius API key for System A
export NEBIUS_API_KEY="your-key-here"

# Install dependencies
pip install -r requirements.txt --break-system-packages
```

### Run Locally

```bash
# Start server
python main.py

# Server starts on http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Test API

```bash
# In another terminal
python test_client.py
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
Returns status of all components.

### Compare Both Systems
```bash
POST /compare
```

**Request:**
```json
{
  "query": "Assess whether this transaction poses elevated risk",
  "transaction": {
    "transaction_id": "TXN_0001",
    "amount": 125000,
    "user_avg_amount": 8200,
    "merchant_category": "Electronics",
    "country": "India",
    "account_age_months": 4
  },
  "run_system_a": true,
  "run_system_b": true
}
```

**Response:**
```json
{
  "experiment_id": "a3b2c1d4",
  "timestamp": "2025-01-29T10:30:00",
  "query": "...",
  "transaction": {...},
  "system_a": {
    "response": "Based on the transaction details...",
    "latency_ms": 423.5,
    "refusal": false,
    "metadata": {
      "model": "deepseek-chat",
      "tokens_used": 245
    }
  },
  "system_b": {
    "response": "This transaction shows elevated risk...",
    "latency_ms": 189.2,
    "route": "transaction_risk",
    "refusal": false,
    "metadata": {
      "model": "DragonLLM/Qwen-Open-Finance-R-8B",
      "handling_mode": "normal"
    }
  }
}
```

### Batch Processing
```bash
POST /batch
```

Process multiple experiments at once.

## 🔧 Configuration

### Environment Variables

```bash
# Required for System A
NEBIUS_API_KEY=your-key-here

# Optional model settings
QWEN_MODEL=DragonLLM/Qwen-Open-Finance-R-8B
QWEN_LOAD_IN_4BIT=true
```

### Model Loading

**DeepSeek V3:**
- No local loading (API-based)
- Instant startup

**Qwen Finance 8B:**
- Downloads ~16GB on first run
- Loads in ~1-2 minutes
- Uses 4-bit quantization to fit in 8GB RAM
- For faster inference, use GPU

## 📊 Running Full Experiments

### Run All 150 Experiments

```python
python test_client.py
# Then uncomment load_experiments_and_run_batch() call
```

This will:
1. Load all experiments from `experiments.jsonl`
2. Run each through both systems
3. Save results to `comparison_results.jsonl`

### Process Results

```python
import pandas as pd

# Load results
df = pd.read_json('comparison_results.jsonl', lines=True)

# Analyze
print(f"Total experiments: {len(df)}")
print(f"Average latency (System A): {df['system_a'].apply(lambda x: x['latency_ms']).mean():.2f}ms")
print(f"Average latency (System B): {df['system_b'].apply(lambda x: x['latency_ms']).mean():.2f}ms")
print(f"Refusal rate (System B): {df['system_b'].apply(lambda x: x['refusal']).sum() / len(df) * 100:.1f}%")
```

## 🐳 Modal Deployment (Recommended)

For production-quality deployment with GPU support:

```python
# modal_app.py
import modal

stub = modal.Stub("finance-comparison")

# Define Qwen image with dependencies
qwen_image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate")
    .run_commands("huggingface-cli download DragonLLM/Qwen-Open-Finance-R-8B")
)

@stub.function(
    image=qwen_image,
    gpu="T4",  # or "A10G" for faster inference
    timeout=300
)
def compare_systems(query, transaction):
    # Your comparison logic here
    pass
```

Deploy:
```bash
modal deploy modal_app.py
```

## ⚡ Performance Tips

### For Local Development
- Use 4-bit quantization (`load_in_4bit=True`)
- Reduce `max_new_tokens` to 256
- Use CPU for Qwen (slower but works)

### For Production (Modal)
- Use GPU (T4 or A10G)
- Load full FP16 model
- Increase `max_new_tokens` to 512
- Use batching for multiple queries

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### Load Test
```bash
# Using locust or similar
locust -f load_test.py
```

## 📝 Logging

Logs are structured JSON:
```json
{
  "timestamp": "2025-01-29T10:30:00",
  "experiment_id": "a3b2c1d4",
  "system": "A",
  "query": "...",
  "response": "...",
  "latency_ms": 423.5,
  "error": null
}
```

## 🚨 Troubleshooting

### "NEBIUS_API_KEY not set"
```bash
export NEBIUS_API_KEY="your-key-here"
```

### "CUDA out of memory"
```python
# Use 4-bit quantization
qwen_engine = QwenFinanceEngine(load_in_4bit=True)
```

### "Model download too slow"
```bash
# Pre-download model
huggingface-cli download DragonLLM/Qwen-Open-Finance-R-8B
```

### "Router not found"
```bash
# Make sure router module is in path
export PYTHONPATH="${PYTHONPATH}:/home/claude/router"
```

## 🎯 For Your Medium Article

This backend demonstrates:
1. ✅ **Clean API design** - REST endpoints for comparison
2. ✅ **Parallel execution** - Both systems run simultaneously
3. ✅ **Behavioral resolution** - Router provides 4 handling modes
4. ✅ **Production-realistic** - FastAPI, error handling, logging

Perfect for showing: *"Here's the system that generated my experimental results"*

## 📈 Next Steps

After backend is working:
1. ✅ Run all 150 experiments
2. ✅ Generate comparison metrics
3. ✅ Create visualizations
4. ✅ Write Medium article

## 🔗 Related Files

- `/router/` - Router v2 implementation
- `/data/` - Synthetic data generation
- `/evaluation/` - Metrics and analysis (to be built)