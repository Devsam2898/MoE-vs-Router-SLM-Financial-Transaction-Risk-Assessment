"""
FastAPI Backend - Main Application

Orchestrates System A (MoE) and System B (Router + SLM) for comparison.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import asyncio
import time
from datetime import datetime
import uuid

# Import our components
import sys
sys.path.append("/home/claude/router")
from router_classifier import Router

# Model clients (will be initialized on startup)
deepseek_client = None
qwen_engine = None
router = None

# Initialize FastAPI app
app = FastAPI(
    title="MoE vs SLM Finance Comparison API",
    description="Compare DeepSeek V3 (MoE) against Qwen-Finance-8B (SLM + Router)",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== MODELS =====

class Transaction(BaseModel):
    """Transaction data structure"""
    transaction_id: str
    amount: float
    user_avg_amount: float
    merchant_category: str
    country: str
    account_age_months: int
    expected_risk_level: Optional[str] = None
    expected_reason: Optional[str] = None


class ComparisonRequest(BaseModel):
    """Request for comparing both systems"""
    query: str = Field(..., description="The user's query about transaction risk")
    transaction: Transaction = Field(..., description="Transaction context")
    run_system_a: bool = Field(default=True, description="Run System A (MoE)?")
    run_system_b: bool = Field(default=True, description="Run System B (SLM)?")


class SystemResult(BaseModel):
    """Result from a single system"""
    response: Optional[str]
    latency_ms: float
    route: Optional[str] = None
    refusal: bool = False
    error: Optional[str] = None
    metadata: Dict = {}


class ComparisonResponse(BaseModel):
    """Response containing results from both systems"""
    experiment_id: str
    timestamp: str
    query: str
    transaction: Transaction
    system_a: Optional[SystemResult] = None
    system_b: Optional[SystemResult] = None


# ===== STARTUP & SHUTDOWN =====

@app.on_event("startup")
async def startup_event():
    """Initialize models and router on startup"""
    global deepseek_client, qwen_engine, router
    
    print("="*60)
    print(" STARTING BACKEND")
    print("="*60)
    
    # Initialize router (lightweight, always load)
    print("\n Loading Router...")
    router = Router()
    print("    Router loaded")
    
    # Initialize DeepSeek client (just API wrapper, no model loading)
    print("\n Initializing DeepSeek client...")
    try:
        from models.deepseek_client import DeepSeekClient
        deepseek_client = DeepSeekClient()
        print("   DeepSeek client ready")
    except Exception as e:
        print(f"    DeepSeek client error: {e}")
        print("   System A will be unavailable")
    
    # Initialize Qwen engine (heavy, loads model)
    print("\n Loading Qwen Finance model...")
    print("   ⏳ This may take 1-2 minutes on first run...")
    try:
        from models.qwen_engine import QwenFinanceEngine
        qwen_engine = QwenFinanceEngine(load_in_4bit=True)
        print("   Qwen engine loaded")
    except Exception as e:
        print(f"    Qwen engine error: {e}")
        print("   System B will use router refusals only")
    
    print("\n" + "="*60)
    print(" BACKEND READY")
    print("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n Shutting down backend...")
    # Cleanup if needed (models will be garbage collected)
    print(" Shutdown complete")


# ===== ENDPOINTS =====

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MoE vs SLM Finance Comparison API",
        "version": "1.0.0",
        "models_loaded": {
            "router": router is not None,
            "deepseek": deepseek_client is not None,
            "qwen": qwen_engine is not None
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "router": "ready" if router else "not loaded",
            "system_a": "ready" if deepseek_client else "not loaded",
            "system_b": "ready" if qwen_engine else "not loaded"
        }
    }


@app.post("/compare", response_model=ComparisonResponse)
async def compare_systems(request: ComparisonRequest):
    """
    Run query through both System A and System B
    
    Returns comparative results for analysis
    """
    experiment_id = str(uuid.uuid4())[:8]
    
    # Run systems in parallel
    system_a_task = run_system_a(request.query, request.transaction.dict()) if request.run_system_a else None
    system_b_task = run_system_b(request.query, request.transaction.dict()) if request.run_system_b else None
    
    # Gather results
    results = await asyncio.gather(
        system_a_task if system_a_task else async_none(),
        system_b_task if system_b_task else async_none(),
        return_exceptions=True
    )
    
    return ComparisonResponse(
        experiment_id=experiment_id,
        timestamp=datetime.now().isoformat(),
        query=request.query,
        transaction=request.transaction,
        system_a=results[0] if request.run_system_a else None,
        system_b=results[1] if request.run_system_b else None
    )


@app.post("/system-a")
async def system_a_only(query: str, transaction: Transaction):
    """Run System A (MoE) only"""
    result = await run_system_a(query, transaction.dict())
    return {"system": "A", "result": result}


@app.post("/system-b")
async def system_b_only(query: str, transaction: Transaction):
    """Run System B (Router + SLM) only"""
    result = await run_system_b(query, transaction.dict())
    return {"system": "B", "result": result}


# ===== SYSTEM EXECUTION =====

async def run_system_a(query: str, transaction: Dict) -> SystemResult:
    """
    Execute System A: DeepSeek V3 (MoE Generalist)
    
    No routing, no refusal logic, just direct inference
    """
    if not deepseek_client:
        return SystemResult(
            response=None,
            latency_ms=0,
            error="System A not available (DeepSeek client not loaded)",
            metadata={}
        )
    
    try:
        # Direct call to MoE
        result = deepseek_client.generate(query, transaction)
        
        return SystemResult(
            response=result["response"],
            latency_ms=result["latency_ms"],
            refusal=False,  # System A never refuses
            error=result["error"],
            metadata={
                "model": result["model"],
                "tokens_used": result.get("tokens_used", 0)
            }
        )
    
    except Exception as e:
        return SystemResult(
            response=None,
            latency_ms=0,
            error=str(e),
            metadata={}
        )


async def run_system_b(query: str, transaction: Dict) -> SystemResult:
    """
    Execute System B: Router + Qwen Finance SLM
    
    1. Classify query with router
    2. Either refuse or call SLM with route-specific prompt
    """
    if not router:
        return SystemResult(
            response=None,
            latency_ms=0,
            error="System B not available (Router not loaded)",
            metadata={}
        )
    
    try:
        # Step 1: Route the query
        route_info = router.classify_query(query)
        
        # Step 2: Handle based on route
        if route_info["should_refuse"]:
            # Complete refusal (no LLM call)
            return SystemResult(
                response=router.get_refusal_message(route_info),
                latency_ms=0.5,  # Router is instant
                route=route_info["route"],
                refusal=True,
                metadata={
                    "router_reason": route_info["metadata"]["reason"],
                    "confidence": route_info["metadata"]["confidence"]
                }
            )
        
        # Step 3: Get handling instructions
        handling = router.get_handling_instructions(route_info)
        
        # Step 4: Call Finance SLM (if loaded)
        if not qwen_engine:
            return SystemResult(
                response=f"[Router classified as {route_info['route']}, but SLM not loaded]",
                latency_ms=0.5,
                route=route_info["route"],
                refusal=False,
                error="Qwen engine not loaded",
                metadata=route_info["metadata"]
            )
        
        result = qwen_engine.generate(
            query=query,
            transaction=transaction,
            route=route_info["route"],
            handling_instruction=handling["instruction"]
        )
        
        return SystemResult(
            response=result["response"],
            latency_ms=result["latency_ms"],
            route=route_info["route"],
            refusal=False,
            error=result["error"],
            metadata={
                "model": result["model"],
                "tokens_generated": result.get("tokens_generated", 0),
                "handling_mode": handling["mode"],
                "router_confidence": route_info["metadata"]["confidence"]
            }
        )
    
    except Exception as e:
        return SystemResult(
            response=None,
            latency_ms=0,
            error=str(e),
            metadata={}
        )


async def async_none():
    """Helper for optional tasks"""
    return None


# ===== BATCH PROCESSING =====

@app.post("/batch")
async def batch_compare(experiments: List[ComparisonRequest]):
    """
    Run multiple experiments in batch
    
    Useful for running all 150 experiments from experiments.jsonl
    """
    results = []
    
    for exp in experiments:
        try:
            result = await compare_systems(exp)
            results.append(result.dict())
        except Exception as e:
            results.append({
                "error": str(e),
                "query": exp.query
            })
    
    return {
        "total": len(experiments),
        "completed": len(results),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print(" STARTING FASTAPI SERVER")
    print("="*60)
    print("\nStarting on http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)