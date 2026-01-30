"""
DeepSeek V3 Client - System A (MoE Generalist)

Wrapper for Nebius Token Factory API to access DeepSeek V3.
"""

import os
import time
from typing import Dict, Optional
import requests
from dotenv import load_dotenv


load_dotenv()

class DeepSeekClient:
    """
    Client for DeepSeek V3 via Nebius Token Factory API
    
    System A: Generalist MoE with no routing, no refusal logic
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize DeepSeek client
        
        Args:
            api_key: Nebius API key (or set NEBIUS_API_KEY env var)
            base_url: API endpoint (default: Nebius Token Factory)
        """
        self.api_key = api_key or os.getenv("NEBIUS_API_KEY")
        if not self.api_key:
            raise ValueError("NEBIUS_API_KEY not set. Get credits at: https://nebius.ai/")
        
        # FIXED: Correct Nebius Token Factory endpoint
        self.base_url = base_url or "https://api.tokenfactory.us-central1.nebius.com/v1"
        
        # FIXED: Correct DeepSeek V3 model name
        self.model = "deepseek-ai/DeepSeek-V3-0324-fast"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(
        self,
        query: str,
        transaction: Dict,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict:
        """
        Generate response using DeepSeek V3
        
        Args:
            query: User's query
            transaction: Transaction context
            temperature: Sampling temperature (0.7 = moderate creativity)
            max_tokens: Maximum response length
        
        Returns:
            Dict with response, latency, and metadata
        """
        # Build prompt (generic financial analyst)
        prompt = self._build_prompt(query, transaction)
        
        # Prepare request
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Time the request
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Extract response
            content = data["choices"][0]["message"]["content"]
            
            return {
                "response": content,
                "latency_ms": round(latency_ms, 2),
                "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                "model": self.model,
                "error": None
            }
        
        except requests.exceptions.Timeout:
            return {
                "response": None,
                "latency_ms": (time.time() - start_time) * 1000,
                "tokens_used": 0,
                "model": self.model,
                "error": "Request timeout (30s)"
            }
        
        except requests.exceptions.RequestException as e:
            return {
                "response": None,
                "latency_ms": (time.time() - start_time) * 1000,
                "tokens_used": 0,
                "model": self.model,
                "error": str(e)
            }
    
    def _build_prompt(self, query: str, transaction: Dict) -> str:
        """
        Build prompt for DeepSeek V3
        
        System A uses a generic financial analyst persona with NO special instructions
        """
        prompt = f"""You are a financial risk analyst. Analyze the following transaction and respond to the query.

Transaction Details:
- Transaction ID: {transaction.get('transaction_id', 'N/A')}
- Amount: ${transaction.get('amount', 0):,}
- User's Average Amount: ${transaction.get('user_avg_amount', 0):,}
- Merchant Category: {transaction.get('merchant_category', 'N/A')}
- Country: {transaction.get('country', 'N/A')}
- Account Age: {transaction.get('account_age_months', 0)} months

Query: {query}

Provide a clear, concise analysis."""

        return prompt


def test_deepseek():
    """Test DeepSeek client"""
    # Check if API key is set
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("⚠️  NEBIUS_API_KEY not set")
        print("   Set it with: export NEBIUS_API_KEY='your-key-here'")
        print("   Get credits at: https://nebius.ai/")
        return
    
    client = DeepSeekClient()
    
    # Test transaction
    transaction = {
        "transaction_id": "TXN_TEST",
        "amount": 125000,
        "user_avg_amount": 8200,
        "merchant_category": "Electronics",
        "country": "India",
        "account_age_months": 4
    }
    
    query = "Assess whether this transaction poses elevated risk."
    
    print("="*60)
    print("🧪 TESTING DEEPSEEK V3 CLIENT")
    print("="*60)
    print(f"Query: {query}")
    print(f"Transaction: {transaction['transaction_id']}")
    print("\nCalling DeepSeek V3...")
    
    result = client.generate(query, transaction)
    
    print("\n" + "="*60)
    print("📊 RESULT")
    print("="*60)
    
    if result["error"]:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Response: {result['response']}")
        print(f"\n📈 Metadata:")
        print(f"   Latency: {result['latency_ms']:.2f}ms")
        print(f"   Tokens: {result['tokens_used']}")
        print(f"   Model: {result['model']}")


if __name__ == "__main__":
    test_deepseek()