"""
Qwen Finance 8B Engine - System B (Domain-Specialized SLM)

Loads and runs Qwen-Open-Finance-R-8B with route-specific prompts.
"""

import os
import time
from typing import Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class QwenFinanceEngine:
    """
    Finance-specialized SLM engine for System B
    
    Uses route-specific system prompts for behavioral control
    """
    
    def __init__(
        self,
        model_name: str = "DragonLLM/Qwen-Open-Finance-R-8B",
        device: str = "auto",
        load_in_4bit: bool = False,
        hf_token: Optional[str] = None
    ):
        """
        Initialize Qwen Finance engine
        
        Args:
            model_name: HuggingFace model identifier
            device: Device placement ("auto", "cuda", "cpu")
            load_in_4bit: Use 4-bit quantization (saves memory)
            hf_token: HuggingFace token (or set HUGGINGFACE_TOKEN env var)
        """
        # Get HuggingFace token from env if not provided
        if hf_token is None:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                raise ValueError(
                    "❌ HUGGINGFACE_TOKEN not set!\n"
                    "   Please add to .env file:\n"
                    "   HUGGINGFACE_TOKEN=hf_your_token_here"
                )
        
        print(f"🔐 Loading Qwen Finance model: {model_name}")
        print(f"   Authenticating with HuggingFace...")
        
        self.model_name = model_name
        self.device = device
        
        # Load tokenizer with authentication
        print(f"   Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token
        )
        
        # Load model with optional quantization
        print(f"   Loading model...")
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "device_map": device,
            "token": hf_token
        }
        
        if load_in_4bit:
            print(f"   Using 4-bit quantization (saves memory)")
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            load_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {self.model.device}")
        print(f"   Memory: {'4-bit' if load_in_4bit else 'FP16'}")
    
    def generate(
        self,
        query: str,
        transaction: Dict,
        route: str,
        handling_instruction: str,
        temperature: float = 0.1,
        max_new_tokens: int = 512
    ) -> Dict:
        """
        Generate response using Qwen Finance model
        
        Args:
            query: User's query
            transaction: Transaction context
            route: Routing category (for logging)
            handling_instruction: Special instruction from router
            temperature: Sampling temperature (0.1 = deterministic)
            max_new_tokens: Maximum response length
        
        Returns:
            Dict with response, latency, and metadata
        """
        # Build prompt with route-specific instructions
        prompt = self._build_prompt(
            query, 
            transaction, 
            handling_instruction
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Time the generation
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
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Decode response (remove input prompt)
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            return {
                "response": response,
                "latency_ms": round(latency_ms, 2),
                "tokens_generated": len(outputs[0]) - len(inputs['input_ids'][0]),
                "model": self.model_name,
                "route": route,
                "error": None
            }
        
        except Exception as e:
            return {
                "response": None,
                "latency_ms": (time.time() - start_time) * 1000,
                "tokens_generated": 0,
                "model": self.model_name,
                "route": route,
                "error": str(e)
            }
    
    def _build_prompt(
        self,
        query: str,
        transaction: Dict,
        handling_instruction: str
    ) -> str:
        """
        Build prompt with route-specific behavioral instructions
        
        This is where System B's discipline comes from!
        """
        # Base system prompt
        base_prompt = """You are a specialized financial risk assessment system. 
Your purpose is to analyze transaction risk based on provided transaction data.

CRITICAL RULES:
- Base your analysis ONLY on the transaction attributes provided
- Use multiple factors in your reasoning
- Be specific and clear
- Do NOT speculate beyond the provided data"""
        
        # Add route-specific instruction
        if handling_instruction:
            base_prompt += f"\n\nSPECIAL INSTRUCTION:\n{handling_instruction}"
        
        # Build full prompt
        prompt = f"""{base_prompt}

Transaction Details:
- Transaction ID: {transaction.get('transaction_id', 'N/A')}
- Amount: ${transaction.get('amount', 0):,}
- User's Average Amount: ${transaction.get('user_avg_amount', 0):,}
- Merchant Category: {transaction.get('merchant_category', 'N/A')}
- Country: {transaction.get('country', 'N/A')}
- Account Age: {transaction.get('account_age_months', 0)} months

Query: {query}

Analysis:"""
        
        return prompt


def test_qwen():
    """Test Qwen Finance engine"""
    print("="*60)
    print("🧪 TESTING QWEN FINANCE ENGINE")
    print("="*60)
    
    # Check for HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("\n❌ ERROR: HUGGINGFACE_TOKEN not set!")
        print("\n📝 To fix this:")
        print("   1. Create a .env file in this directory")
        print("   2. Add this line:")
        print("      HUGGINGFACE_TOKEN=hf_your_token_here")
        print("\n🔑 Get your token:")
        print("   https://huggingface.co/settings/tokens")
        print("\n✅ Request model access:")
        print("   https://huggingface.co/DragonLLM/Qwen-Open-Finance-R-8B")
        return
    
    print(f"\n✅ HuggingFace token found: {hf_token[:10]}...")
    print(f"\n⚠️  Note: First run will download ~16GB model")
    print(f"   Make sure you have enough disk space and RAM\n")
    
    try:
        # Initialize engine (use 4-bit to save memory)
        engine = QwenFinanceEngine(load_in_4bit=True)
        
        # Test transaction
        transaction = {
            "transaction_id": "TXN_TEST",
            "amount": 125000,
            "user_avg_amount": 8200,
            "merchant_category": "Electronics",
            "country": "India",
            "account_age_months": 4
        }
        
        # Test with normal mode
        query = "Assess whether this transaction poses elevated risk."
        instruction = "Analyze transaction risk based on provided data. Use multiple factors."
        
        print(f"\n📋 Test Query:")
        print(f"   {query}")
        print(f"\n🔀 Route: transaction_risk")
        print(f"\n⏳ Generating response...\n")
        
        result = engine.generate(
            query=query,
            transaction=transaction,
            route="transaction_risk",
            handling_instruction=instruction
        )
        
        print("\n" + "="*60)
        print("📊 RESULT")
        print("="*60)
        
        if result["error"]:
            print(f"\n❌ Error: {result['error']}")
        else:
            print(f"\n✅ Response:")
            print(f"{result['response']}")
            print(f"\n📈 Metadata:")
            print(f"   Latency: {result['latency_ms']:.2f}ms")
            print(f"   Tokens Generated: {result['tokens_generated']}")
            print(f"   Model: {result['model']}")
            print(f"   Route: {result['route']}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check RAM: Need 16GB+ for full model")
        print("   2. Install: pip install transformers torch accelerate bitsandbytes")
        print("   3. Request model access on HuggingFace")
        print("   4. Use load_in_4bit=True to reduce memory")


if __name__ == "__main__":
    test_qwen()



# """
# Qwen Finance 8B Engine - System B (Domain-Specialized SLM)

# Loads and runs Qwen-Open-Finance-R-8B with route-specific prompts.
# """

# import time
# from typing import Dict, Optional
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer


# class QwenFinanceEngine:
#     """
#     Finance-specialized SLM engine for System B
    
#     Uses route-specific system prompts for behavioral control
#     """
    
#     def __init__(
#         self,
#         model_name: str = "DragonLLM/Qwen-Open-Finance-R-8B",
#         device: str = "auto",
#         load_in_4bit: bool = False
#     ):
#         """
#         Initialize Qwen Finance engine
        
#         Args:
#             model_name: HuggingFace model identifier
#             device: Device placement ("auto", "cuda", "cpu")
#             load_in_4bit: Use 4-bit quantization (saves memory)
#         """
#         print(f" Loading Qwen Finance model: {model_name}")
        
#         self.model_name = model_name
#         self.device = device
        
#         # Load tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             trust_remote_code=True
#         )
        
#         # Load model with optional quantization
#         load_kwargs = {
#             "trust_remote_code": True,
#             "torch_dtype": torch.float16,
#             "device_map": device
#         }
        
#         if load_in_4bit:
#             load_kwargs["load_in_4bit"] = True
#             load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
        
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             **load_kwargs
#         )
        
#         print(f" Model loaded on: {self.model.device}")
    
#     def generate(
#         self,
#         query: str,
#         transaction: Dict,
#         route: str,
#         handling_instruction: str,
#         temperature: float = 0.1,
#         max_new_tokens: int = 512
#     ) -> Dict:
#         """
#         Generate response using Qwen Finance model
        
#         Args:
#             query: User's query
#             transaction: Transaction context
#             route: Routing category (for logging)
#             handling_instruction: Special instruction from router
#             temperature: Sampling temperature (0.1 = deterministic)
#             max_new_tokens: Maximum response length
        
#         Returns:
#             Dict with response, latency, and metadata
#         """
#         # Build prompt with route-specific instructions
#         prompt = self._build_prompt(
#             query, 
#             transaction, 
#             handling_instruction
#         )
        
#         # Tokenize
#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
#         # Time the generation
#         start_time = time.time()
        
#         try:
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     **inputs,
#                     max_new_tokens=max_new_tokens,
#                     temperature=temperature,
#                     do_sample=temperature > 0,
#                     pad_token_id=self.tokenizer.eos_token_id
#                 )
            
#             end_time = time.time()
#             latency_ms = (end_time - start_time) * 1000
            
#             # Decode response (remove input prompt)
#             full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#             response = full_response[len(prompt):].strip()
            
#             return {
#                 "response": response,
#                 "latency_ms": round(latency_ms, 2),
#                 "tokens_generated": len(outputs[0]) - len(inputs['input_ids'][0]),
#                 "model": self.model_name,
#                 "route": route,
#                 "error": None
#             }
        
#         except Exception as e:
#             return {
#                 "response": None,
#                 "latency_ms": (time.time() - start_time) * 1000,
#                 "tokens_generated": 0,
#                 "model": self.model_name,
#                 "route": route,
#                 "error": str(e)
#             }
    
#     def _build_prompt(
#         self,
#         query: str,
#         transaction: Dict,
#         handling_instruction: str
#     ) -> str:
#         """
#         Build prompt with route-specific behavioral instructions
        
#         This is where System B's discipline comes from!
#         """
#         # Base system prompt
#         base_prompt = """You are a specialized financial risk assessment system. 
# Your purpose is to analyze transaction risk based on provided transaction data.

# CRITICAL RULES:
# - Base your analysis ONLY on the transaction attributes provided
# - Use multiple factors in your reasoning
# - Be specific and clear
# - Do NOT speculate beyond the provided data"""
        
#         # Add route-specific instruction
#         if handling_instruction:
#             base_prompt += f"\n\nSPECIAL INSTRUCTION:\n{handling_instruction}"
        
#         # Build full prompt
#         prompt = f"""{base_prompt}

# Transaction Details:
# - Transaction ID: {transaction.get('transaction_id', 'N/A')}
# - Amount: ${transaction.get('amount', 0):,}
# - User's Average Amount: ${transaction.get('user_avg_amount', 0):,}
# - Merchant Category: {transaction.get('merchant_category', 'N/A')}
# - Country: {transaction.get('country', 'N/A')}
# - Account Age: {transaction.get('account_age_months', 0)} months

# Query: {query}

# Analysis:"""
        
#         return prompt


# def test_qwen():
#     """Test Qwen Finance engine"""
#     print("="*60)
#     print(" TESTING QWEN FINANCE ENGINE")
#     print("="*60)
#     print("\n  Note: This will download ~16GB model on first run")
#     print("   Make sure you have enough disk space and RAM")
    
#     try:
#         # Initialize engine
#         engine = QwenFinanceEngine(load_in_4bit=True)  # Use 4-bit to save memory
        
#         # Test transaction
#         transaction = {
#             "transaction_id": "TXN_TEST",
#             "amount": 125000,
#             "user_avg_amount": 8200,
#             "merchant_category": "Electronics",
#             "country": "India",
#             "account_age_months": 4
#         }
        
#         # Test with normal mode
#         query = "Assess whether this transaction poses elevated risk."
#         instruction = "Analyze transaction risk based on provided data. Use multiple factors."
        
#         print(f"\n Query: {query}")
#         print(f" Route: transaction_risk")
#         print(f"\nGenerating response...")
        
#         result = engine.generate(
#             query=query,
#             transaction=transaction,
#             route="transaction_risk",
#             handling_instruction=instruction
#         )
        
#         print("\n" + "="*60)
#         print(" RESULT")
#         print("="*60)
        
#         if result["error"]:
#             print(f" Error: {result['error']}")
#         else:
#             print(f" Response:\n{result['response']}")
#             print(f"\n Metadata:")
#             print(f"   Latency: {result['latency_ms']:.2f}ms")
#             print(f"   Tokens Generated: {result['tokens_generated']}")
#             print(f"   Model: {result['model']}")
    
#     except Exception as e:
#         print(f"\n Error loading model: {e}")
#         print("\nTroubleshooting:")
#         print("1. Make sure you have enough RAM (16GB+ recommended)")
#         print("2. Install dependencies: pip install transformers torch accelerate --break-system-packages")
#         print("3. For Modal deployment, this will work fine with GPU")


# if __name__ == "__main__":
#     test_qwen()