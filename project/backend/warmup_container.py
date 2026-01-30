"""
Warm Up Modal Container

Run this FIRST to load models before running experiments.
The first request takes 30-60 seconds, so we do it separately.
"""

import requests
import time

# Your Modal URL
MODAL_URL = "https://devsam2898--finance-comparison-api-fastapi-app.modal.run"

def warm_up():
    """Send a test request to warm up the container"""
    print("="*60)
    print("🔥 WARMING UP MODAL CONTAINER")
    print("="*60)
    print("\nThis will take 30-60 seconds on first run...")
    print("(Loading Qwen model ~16GB into GPU memory)")
    print()
    
    # Test transaction
    test_data = {
        "query": "Assess whether this transaction poses elevated risk",
        "transaction": {
            "transaction_id": "TXN_WARMUP",
            "amount": 125000,
            "user_avg_amount": 8200,
            "merchant_category": "Electronics",
            "country": "India",
            "account_age_months": 4
        }
    }
    
    start = time.time()
    
    try:
        print("⏳ Sending warmup request...")
        response = requests.post(
            f"{MODAL_URL}/compare",
            json=test_data,
            timeout=360  # 6 minutes max
        )
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Container warmed up in {elapsed:.1f}s")
            print(f"\n📊 Test Result:")
            
            if result.get("system_a"):
                print(f"\n   System A:")
                print(f"   - Latency: {result['system_a']['latency_ms']:.0f}ms")
                print(f"   - Response: {result['system_a']['response'][:80]}...")
            
            if result.get("system_b"):
                print(f"\n   System B:")
                print(f"   - Route: {result['system_b']['route']}")
                print(f"   - Latency: {result['system_b']['latency_ms']:.0f}ms")
                print(f"   - Response: {result['system_b']['response'][:80]}...")
            
            print(f"\n✅ Ready to run experiments!")
            print(f"   Next requests will be fast (~2-3s)")
            return True
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            return False
    
    except requests.exceptions.Timeout:
        print(f"\n⏱️  Timeout after {time.time() - start:.1f}s")
        print("\n💡 This might mean:")
        print("   1. Model is still loading (wait and try again)")
        print("   2. Container ran out of memory")
        print("   3. Network issue")
        print("\nCheck Modal logs:")
        print("   modal app logs finance-comparison-api")
        return False
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nCheck Modal logs:")
        print("   modal app logs finance-comparison-api")
        return False


if __name__ == "__main__":
    print("\n🎯 This script warms up the Modal container")
    print("Run this BEFORE running experiments to avoid timeouts\n")
    
    success = warm_up()
    
    if success:
        print("\n" + "="*60)
        print("🚀 NOW YOU CAN RUN EXPERIMENTS")
        print("="*60)
        print("\npython run_experiments_modal.py")
    else:
        print("\n" + "="*60)
        print("⚠️  WARMUP FAILED")
        print("="*60)
        print("\nTroubleshooting:")
        print("1. Wait 2-3 minutes for container to fully start")
        print("2. Try again: python warmup.py")
        print("3. Check logs: modal app logs finance-comparison-api")