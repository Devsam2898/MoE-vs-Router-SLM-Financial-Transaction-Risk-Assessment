"""
Modal Deployment - Finance Comparison API (FIXED for gated models)

Handles HuggingFace authentication for gated models.
"""

import modal

# Create Modal app
app = modal.App("finance-comparison-api")

# Define image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.109.0",
        "pydantic==2.5.3",
        "requests==2.31.0",
        "openai==1.58.1",         # For DeepSeek V3 via Nebius
        "torch==2.10.0",           # Latest PyTorch
        "transformers==5.0.0",     # Qwen3 support!
        "accelerate==1.12.0",      # Latest accelerate
        "bitsandbytes==0.49.1",    # For 4-bit quantization
        "huggingface-hub",
    )
)

# Secrets - need BOTH Nebius and HuggingFace
secrets = [
    modal.Secret.from_name("nebius-api-key"),      # Your Nebius API key
    modal.Secret.from_name("huggingface-key")  # Your HuggingFace token
]

# GPU configuration (A10G for faster inference)
GPU_CONFIG = modal.gpu.A10G()  # 24GB VRAM, 2x faster than T4

# ===== IMPORTS =====

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import asyncio
from datetime import datetime
import uuid


# ===== PYDANTIC MODELS =====

class Transaction(BaseModel):
    transaction_id: str
    amount: float
    user_avg_amount: float
    merchant_category: str
    country: str
    account_age_months: int
    expected_risk_level: Optional[str] = None
    expected_reason: Optional[str] = None


class ComparisonRequest(BaseModel):
    query: str = Field(..., description="The user's query")
    transaction: Transaction = Field(..., description="Transaction context")
    run_system_a: bool = Field(default=True)
    run_system_b: bool = Field(default=True)


class SystemResult(BaseModel):
    response: Optional[str]
    latency_ms: float
    route: Optional[str] = None
    refusal: bool = False
    error: Optional[str] = None
    metadata: Dict = {}


class ComparisonResponse(BaseModel):
    experiment_id: str
    timestamp: str
    query: str
    transaction: Transaction
    system_a: Optional[SystemResult] = None
    system_b: Optional[SystemResult] = None


# ===== MODAL FUNCTION =====

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    secrets=secrets,
    timeout=600,
    scaledown_window=300,  # Updated from container_idle_timeout
)
@modal.concurrent(max_inputs=10)  # Updated from allow_concurrent_inputs
@modal.asgi_app()
def fastapi_app():
    """Modal entry point - wraps FastAPI app"""
    import os
    import sys
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import requests
    
    # Initialize FastAPI
    web_app = FastAPI(
        title="Finance Comparison API",
        description="Compare MoE vs SLM",
        version="1.0.0"
    )
    
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ===== MODEL INITIALIZATION (on container startup) =====
    
    deepseek_client = None
    qwen_engine = None
    router = None
    
    print("🚀 Initializing models...")
    
    # 1. Router (simple inline version)
    print("1️⃣ Loading Router...")
    
    class SimpleRouter:
        def classify_query(self, query: str) -> Dict:
            query_lower = query.lower()
            
            # Out-of-scope keywords
            out_of_scope = ["predict", "forecast", "demonetization", "rbi guidelines", 
                           "machine learning", "global fraud rate", "sebi"]
            
            # Mixed intent markers
            mixed_markers = ["and also", "also tell me", "along with"]
            
            # Check for refusal triggers
            for keyword in ["predict", "forecast", "should i invest"]:
                if keyword in query_lower:
                    return {
                        "route": "explicit_refusal",
                        "should_refuse": True,
                        "metadata": {"reason": f"Contains: {keyword}", "confidence": "high"}
                    }
            
            # Check for governance
            for keyword in ["rbi", "sebi", "regulation"]:
                if keyword in query_lower:
                    return {
                        "route": "governance",
                        "should_refuse": False,
                        "metadata": {"reason": "Governance question", "confidence": "high"}
                    }
            
            # Check for mixed intent
            has_transaction = any(w in query_lower for w in ["risk", "assess", "transaction"])
            has_out_of_scope = any(w in query_lower for w in out_of_scope)
            
            if has_transaction and has_out_of_scope:
                return {
                    "route": "mixed_intent",
                    "should_refuse": False,
                    "metadata": {"reason": "Mixed intent", "confidence": "high"}
                }
            
            # Default: transaction risk
            return {
                "route": "transaction_risk",
                "should_refuse": False,
                "metadata": {"reason": "Transaction risk analysis", "confidence": "default"}
            }
        
        def get_refusal_message(self, route_info: Dict) -> str:
            return "I cannot assist with this request. I am designed to analyze transaction risk based on provided data."
        
        def get_handling_instructions(self, route_info: Dict) -> Dict:
            route = route_info['route']
            
            if route == "transaction_risk":
                return {
                    "mode": "normal",
                    "instruction": "Analyze transaction risk based on provided data. Use multiple factors."
                }
            elif route == "governance":
                return {
                    "mode": "conservative",
                    "instruction": "Focus ONLY on transaction attributes. Do NOT speculate about regulations."
                }
            elif route == "mixed_intent":
                return {
                    "mode": "partial",
                    "instruction": "Answer ONLY the transaction risk part. Refuse out-of-scope elements explicitly."
                }
            else:
                return {"mode": "refuse", "instruction": None}
    
    router = SimpleRouter()
    print("   ✅ Router loaded")
    
    # 2. DeepSeek Client
    print("2️⃣ Initializing DeepSeek client...")
    
    class DeepSeekClient:
        def __init__(self):
            self.api_key = os.getenv("NEBIUS_API_KEY")
            if not self.api_key:
                raise ValueError("NEBIUS_API_KEY not set")
            
            # Use OpenAI client with Nebius Token Factory endpoint
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
                api_key=self.api_key
            )
            self.model = "deepseek-ai/DeepSeek-V3-0324-fast"
        
        def generate(self, query: str, transaction: Dict, temperature: float = 0.7, max_tokens: int = 512) -> Dict:
            system_prompt = "You are a financial risk analyst. Analyze the transaction and respond to the query."
            
            user_message = f"""Transaction Details:
- Amount: ${transaction.get('amount', 0):,}
- User Average: ${transaction.get('user_avg_amount', 0):,}
- Merchant: {transaction.get('merchant_category', 'N/A')}
- Country: {transaction.get('country', 'N/A')}
- Account Age: {transaction.get('account_age_months', 0)} months

Query: {query}

Provide a clear analysis."""
            
            start_time = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                latency_ms = (time.time() - start_time) * 1000
                
                return {
                    "response": response.choices[0].message.content,
                    "latency_ms": round(latency_ms, 2),
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "model": self.model,
                    "error": None
                }
            except Exception as e:
                return {
                    "response": None,
                    "latency_ms": (time.time() - start_time) * 1000,
                    "tokens_used": 0,
                    "model": self.model,
                    "error": str(e)
                }
    
    try:
        deepseek_client = DeepSeekClient()
        print("   ✅ DeepSeek client ready")
    except Exception as e:
        print(f"   ⚠️  DeepSeek error: {e}")
    
    # 3. Qwen Engine (loads model with HF authentication)
    print("3️⃣ Loading Qwen Finance model...")
    
    class QwenEngine:
        def __init__(self):
            # Using Qwen3 finance-specialized model (supported in transformers 5.0.0+)
            model_name = "DragonLLM/Qwen-Open-Finance-R-8B"
            
            # Use HuggingFace token for authentication
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                raise ValueError("HUGGINGFACE_TOKEN not set")
            
            print(f"   Downloading {model_name} with authentication...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token
            )
            
            # Check if we should use 4-bit quantization (can disable if problematic)
            use_4bit = os.getenv("USE_4BIT_QUANT", "true").lower() == "true"
            
            if use_4bit:
                print(f"   Loading with 4-bit quantization...")
                # Configure 4-bit quantization properly
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    quantization_config=quantization_config,
                    token=hf_token
                )
            else:
                print(f"   Loading in FP16 (no quantization)...")
                # Load in FP16 without quantization (uses more memory but more stable)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=hf_token
                )
            
            self.model_name = model_name
            print(f"   ✅ Model loaded on: {self.model.device}")
        
        def generate(self, query: str, transaction: Dict, route: str,
                    handling_instruction: str, temperature: float = 0.1,
                    max_new_tokens: int = 512) -> Dict:
            
            base_prompt = """You are a specialized financial risk assessment system.
Analyze transaction risk based ONLY on provided data.

CRITICAL RULES:
- Use multiple factors in your reasoning
- Be specific and clear
- Do NOT speculate beyond provided data"""
            
            if handling_instruction:
                base_prompt += f"\n\nSPECIAL INSTRUCTION:\n{handling_instruction}"
            
            prompt = f"""{base_prompt}

Transaction Details:
- Amount: ${transaction.get('amount', 0):,}
- User Average: ${transaction.get('user_avg_amount', 0):,}
- Merchant: {transaction.get('merchant_category', 'N/A')}
- Country: {transaction.get('country', 'N/A')}
- Account Age: {transaction.get('account_age_months', 0)} months

Query: {query}

Analysis:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            start_time = time.time()
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = full_response[len(prompt):].strip()
                
                return {
                    "response": response,
                    "latency_ms": round((time.time() - start_time) * 1000, 2),
                    "tokens_generated": len(outputs[0]) - len(inputs['input_ids'][0]),
                    "model": self.model_name,
                    "route": route,
                    "error": None
                }
            except Exception as e:
                return {
                    "response": None,
                    "latency_ms": round((time.time() - start_time) * 1000, 2),
                    "tokens_generated": 0,
                    "model": self.model_name,
                    "route": route,
                    "error": str(e)
                }
    
    try:
        qwen_engine = QwenEngine()
        print("   ✅ Qwen engine loaded")
    except Exception as e:
        print(f"   ⚠️  Qwen error: {e}")
    
    print("✅ All models initialized!")
    
    # ===== ENDPOINTS =====
    
    @web_app.get("/")
    async def root():
        return {
            "status": "healthy",
            "service": "Finance Comparison API",
            "environment": "modal",
            "models_loaded": {
                "router": router is not None,
                "deepseek": deepseek_client is not None,
                "qwen": qwen_engine is not None
            }
        }
    
    @web_app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "router": "ready" if router else "not loaded",
                "system_a": "ready" if deepseek_client else "not loaded",
                "system_b": "ready" if qwen_engine else "not loaded"
            }
        }
    
    @web_app.post("/compare", response_model=ComparisonResponse)
    async def compare_systems(request: ComparisonRequest):
        """Run query through both systems"""
        experiment_id = str(uuid.uuid4())[:8]
        
        async def run_system_a(query: str, transaction: Dict):
            if not deepseek_client:
                return SystemResult(response=None, latency_ms=0, error="DeepSeek not available", metadata={})
            try:
                result = deepseek_client.generate(query, transaction)
                return SystemResult(
                    response=result["response"],
                    latency_ms=result["latency_ms"],
                    refusal=False,
                    error=result["error"],
                    metadata={"model": result["model"], "tokens_used": result.get("tokens_used", 0)}
                )
            except Exception as e:
                return SystemResult(response=None, latency_ms=0, error=str(e), metadata={})
        
        async def run_system_b(query: str, transaction: Dict):
            if not router:
                return SystemResult(response=None, latency_ms=0, error="Router not available", metadata={})
            try:
                route_info = router.classify_query(query)
                
                if route_info["should_refuse"]:
                    return SystemResult(
                        response=router.get_refusal_message(route_info),
                        latency_ms=0.5,
                        route=route_info["route"],
                        refusal=True,
                        metadata=route_info["metadata"]
                    )
                
                handling = router.get_handling_instructions(route_info)
                
                if not qwen_engine:
                    return SystemResult(
                        response=f"[Router: {route_info['route']}, SLM not loaded]",
                        latency_ms=0.5,
                        route=route_info["route"],
                        error="Qwen not loaded",
                        metadata=route_info["metadata"]
                    )
                
                result = qwen_engine.generate(query, transaction, route_info["route"], handling["instruction"])
                
                return SystemResult(
                    response=result["response"],
                    latency_ms=result["latency_ms"],
                    route=route_info["route"],
                    refusal=False,
                    error=result["error"],
                    metadata={
                        "model": result["model"],
                        "tokens_generated": result.get("tokens_generated", 0),
                        "handling_mode": handling["mode"]
                    }
                )
            except Exception as e:
                return SystemResult(response=None, latency_ms=0, error=str(e), metadata={})
        
        # Run in parallel
        results = await asyncio.gather(
            run_system_a(request.query, request.transaction.model_dump()) if request.run_system_a else asyncio.sleep(0),
            run_system_b(request.query, request.transaction.model_dump()) if request.run_system_b else asyncio.sleep(0),
            return_exceptions=True
        )
        
        return ComparisonResponse(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            query=request.query,
            transaction=request.transaction,
            system_a=results[0] if request.run_system_a and isinstance(results[0], SystemResult) else None,
            system_b=results[1] if request.run_system_b and isinstance(results[1], SystemResult) else None
        )
    
    @web_app.post("/batch")
    async def batch_compare(experiments: List[ComparisonRequest]):
        """Run multiple experiments"""
        results = []
        for exp in experiments:
            try:
                result = await compare_systems(exp)
                results.append(result.dict())
            except Exception as e:
                results.append({"error": str(e), "query": exp.query})
        return {"total": len(experiments), "completed": len(results), "results": results}
    
    return web_app


if __name__ == "__main__":
    print("Deploy with: modal deploy modal_app.py")