"""
Run All Experiments on Modal Deployment

This script runs all 150 experiments through your deployed Modal API
and saves results for analysis.
"""

import requests
import json
import time
from tqdm import tqdm
from datetime import datetime

# ===== CONFIGURATION =====

# Your Modal deployment URL (replace after deploying)
MODAL_URL = "https://devsam2898--finance-comparison-api-fastapi-app.modal.run"

# Paths
EXPERIMENTS_FILE = r"D:\\Financial Assistant for nomad shareholders\\SLM, MOE in Finance\\experiments.jsonl"
OUTPUT_FILE = "modal_results.jsonl"

# Request settings
TIMEOUT = 60  # seconds
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds


# ===== FUNCTIONS =====

def load_experiments(filepath: str) -> list:
    """Load experiments from JSONL file"""
    experiments = []
    with open(filepath, 'r') as f:
        for line in f:
            exp = json.loads(line)
            experiments.append({
                "query": exp["query"],
                "transaction": exp["transaction"],
                "category": exp.get("category", "unknown"),
                "experiment_id": exp.get("experiment_id", "unknown")
            })
    return experiments


def run_single_experiment(exp: dict, attempt: int = 1) -> dict:
    """Run a single experiment with retry logic"""
    try:
        response = requests.post(
            f"{MODAL_URL}/compare",
            json={
                "query": exp["query"],
                "transaction": exp["transaction"]
            },
            timeout=TIMEOUT
        )
        response.raise_for_status()
        
        result = response.json()
        # Add original experiment metadata
        result["original_experiment_id"] = exp["experiment_id"]
        result["category"] = exp["category"]
        
        return result
    
    except requests.exceptions.Timeout:
        if attempt < RETRY_ATTEMPTS:
            print(f"\n Timeout, retrying ({attempt}/{RETRY_ATTEMPTS})...")
            time.sleep(RETRY_DELAY)
            return run_single_experiment(exp, attempt + 1)
        else:
            return {
                "error": "Timeout after retries",
                "query": exp["query"],
                "experiment_id": exp["experiment_id"]
            }
    
    except requests.exceptions.RequestException as e:
        if attempt < RETRY_ATTEMPTS:
            print(f"\n Request failed, retrying ({attempt}/{RETRY_ATTEMPTS})...")
            time.sleep(RETRY_DELAY)
            return run_single_experiment(exp, attempt + 1)
        else:
            return {
                "error": str(e),
                "query": exp["query"],
                "experiment_id": exp["experiment_id"]
            }


def run_all_experiments():
    """Run all experiments and save results"""
    print("="*60)
    print(" RUNNING ALL EXPERIMENTS ON MODAL")
    print("="*60)
    
    # Check if URL is configured
    if "your-username" in MODAL_URL:
        print("\n  ERROR: Please update MODAL_URL in this script!")
        print("   Replace 'your-username' with your actual Modal username")
        print("   Get URL from: modal deploy modal_app.py")
        return
    
    # Health check
    print("\n Health Check...")
    try:
        response = requests.get(f"{MODAL_URL}/health", timeout=10)
        response.raise_for_status()
        health = response.json()
        print(f"   API is healthy")
        print(f"   Components: {health['components']}")
    except Exception as e:
        print(f"   Health check failed: {e}")
        print("   Make sure Modal app is deployed and URL is correct")
        return
    
    # Load experiments
    print("\n Loading Experiments...")
    experiments = load_experiments(EXPERIMENTS_FILE)
    print(f"   Loaded {len(experiments)} experiments")
    
    # Run experiments
    print("\n Running Experiments...")
    print(f"   Total: {len(experiments)}")
    print(f"   Timeout: {TIMEOUT}s per request")
    print(f"   Retries: {RETRY_ATTEMPTS} attempts")
    print()
    
    results = []
    errors = 0
    start_time = time.time()
    
    for exp in tqdm(experiments, desc="Progress"):
        result = run_single_experiment(exp)
        results.append(result)
        
        if "error" in result:
            errors += 1
    
    elapsed = time.time() - start_time
    
    # Save results
    print("\n4️⃣ Saving Results...")
    with open(OUTPUT_FILE, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"    Saved to: {OUTPUT_FILE}")
    
    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    print(f"Total Experiments: {len(results)}")
    print(f"Successful: {len(results) - errors}")
    print(f"Errors: {errors}")
    print(f"Time Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Avg Time per Experiment: {elapsed/len(results):.2f}s")
    
    # Compute basic stats (if no errors)
    if errors == 0:
        print("\n Performance Stats:")
        
        # System A latency
        system_a_latencies = [r["system_a"]["latency_ms"] for r in results if r.get("system_a")]
        if system_a_latencies:
            print(f"   System A Avg Latency: {sum(system_a_latencies)/len(system_a_latencies):.2f}ms")
        
        # System B latency
        system_b_latencies = [r["system_b"]["latency_ms"] for r in results if r.get("system_b")]
        if system_b_latencies:
            print(f"   System B Avg Latency: {sum(system_b_latencies)/len(system_b_latencies):.2f}ms")
        
        # System B routes
        routes = [r["system_b"].get("route", "unknown") for r in results if r.get("system_b")]
        route_counts = {}
        for route in routes:
            route_counts[route] = route_counts.get(route, 0) + 1
        
        print(f"\n System B Routes:")
        for route, count in sorted(route_counts.items()):
            print(f"   {route}: {count} ({count/len(routes)*100:.1f}%)")
    
    print("\n Done! Results saved to:", OUTPUT_FILE)
    print("\nNext steps:")
    print("1. Analyze results: python ../evaluation/analyze.py")
    print("2. Generate visualizations")
    print("3. Write Medium article")


def test_single():
    """Test with a single experiment"""
    print("="*60)
    print(" TESTING SINGLE EXPERIMENT")
    print("="*60)
    
    # Load one experiment
    experiments = load_experiments(EXPERIMENTS_FILE)
    exp = experiments[0]
    
    print(f"\nQuery: {exp['query']}")
    print(f"Category: {exp['category']}")
    print("\nSending request...")
    
    result = run_single_experiment(exp)
    
    print("\n" + "="*60)
    print(" RESULT")
    print("="*60)
    
    if "error" in result:
        print(f" Error: {result['error']}")
    else:
        print(f" Success!")
        print(f"\nExperiment ID: {result['experiment_id']}")
        
        if result.get("system_a"):
            print(f"\n SYSTEM A:")
            print(f"   Latency: {result['system_a']['latency_ms']:.2f}ms")
            print(f"   Response: {result['system_a']['response'][:100]}...")
        
        if result.get("system_b"):
            print(f"\n SYSTEM B:")
            print(f"   Route: {result['system_b']['route']}")
            print(f"   Latency: {result['system_b']['latency_ms']:.2f}ms")
            print(f"   Refusal: {result['system_b']['refusal']}")
            print(f"   Response: {result['system_b']['response'][:100]}...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode: run single experiment
        test_single()
    else:
        # Production mode: run all experiments
        run_all_experiments()