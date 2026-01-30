"""
Router - Main Interface for Query Classification

This is the main router that System B uses to decide how to handle
each query. It wraps the rule-based classifier and provides logging.
"""

from enum import Enum
from typing import Dict, Optional, Optional
from router_rules import RouteClassifier


class Route(str, Enum):
    """Enumeration of available routing categories"""
    TRANSACTION_RISK = "transaction_risk"
    MIXED_INTENT = "mixed_intent"
    GOVERNANCE = "governance"
    EXPLICIT_REFUSAL = "explicit_refusal"


class Router:
    """
    Main router for System B
    
    Classifies queries and determines appropriate handling strategy.
    """
    
    def __init__(self):
        self.classifier = RouteClassifier()
        self.classification_count = 0
    
    def classify_query(self, query: str) -> Dict:
        """
        Classify a query and return routing decision
        
        Args:
            query: User's query string
        
        Returns:
            Dictionary containing:
            - route: The routing category (Route enum value)
            - metadata: Classification details
            - should_refuse: Boolean indicating if query should be refused
        """
        self.classification_count += 1
        
        # Get classification from rule-based classifier
        route_str, metadata = self.classifier.classify(query)
        
        # Convert to enum
        route = Route(route_str)
        
        # Determine handling strategy
        # Only EXPLICIT_REFUSAL means complete refusal
        # Others are processed, but with different system prompts
        should_refuse = (route == Route.EXPLICIT_REFUSAL)
        requires_conservative_answer = (route == Route.GOVERNANCE)
        requires_partial_refusal = (route == Route.MIXED_INTENT)
        
        return {
            "route": route.value,
            "metadata": metadata,
            "should_refuse": should_refuse,
            "conservative_mode": requires_conservative_answer,
            "partial_refusal_mode": requires_partial_refusal,
            "classification_id": self.classification_count
        }
    
    def get_refusal_message(self, route_info: Dict) -> Optional[str]:
        """
        Generate appropriate refusal message for explicit refusals only
        """
        if not route_info["should_refuse"]:
            return None
        
        reason = route_info["metadata"].get("reason", "Query is out of scope")
        
        return (
            "I cannot assist with this request. "
            "I am designed to analyze transaction risk based on provided transaction data. "
            f"This query is out of scope: {reason}. "
            "Please rephrase your question to focus on the specific transaction risk assessment."
        )
    
    def get_handling_instructions(self, route_info: Dict) -> Dict:
        """
        Get system prompt modifications based on route
        
        Returns instructions for how the Finance SLM should handle this query
        """
        route = route_info['route']
        
        if route == Route.EXPLICIT_REFUSAL.value:
            return {
                "mode": "refuse",
                "instruction": None  # Don't call LLM
            }
        
        elif route == Route.GOVERNANCE.value:
            return {
                "mode": "conservative",
                "instruction": (
                    "This query mentions regulatory/governance topics. "
                    "Focus ONLY on observable transaction attributes. "
                    "Do NOT speculate about regulations, compliance requirements, or legal matters. "
                    "If asked about regulations, state: 'I cannot comment on regulatory requirements.'"
                )
            }
        
        elif route == Route.MIXED_INTENT.value:
            return {
                "mode": "partial",
                "instruction": (
                    "This query has multiple parts. "
                    "Answer ONLY the transaction risk assessment part. "
                    "For any out-of-scope components (macro trends, ML explanations, etc.), "
                    "explicitly state: 'I can only assess the specific transaction risk based on provided data.'"
                )
            }
        
        else:  # TRANSACTION_RISK
            return {
                "mode": "normal",
                "instruction": (
                    "Analyze transaction risk based on provided data. "
                    "Use multiple factors in your assessment. "
                    "Be specific and clear in your reasoning."
                )
            }
    
    def explain_routing(self, query: str) -> str:
        """Generate human-readable explanation of routing decision"""
        route_info = self.classify_query(query)
        
        explanation = [
            f"Query: {query}",
            f"Route: {route_info['route']}",
            f"Should Refuse: {route_info['should_refuse']}",
            f"Confidence: {route_info['metadata']['confidence']}",
            f"Reason: {route_info['metadata']['reason']}"
        ]
        
        if route_info['should_refuse']:
            explanation.append(f"\nRefusal Message:\n{self.get_refusal_message(route_info)}")
        
        return "\n".join(explanation)
    
    def get_stats(self) -> Dict:
        """Return routing statistics"""
        return {
            "total_classifications": self.classification_count
        }


def demo():
    """Interactive demo of the router"""
    router = Router()
    
    print("="*70)
    print(" ROUTER DEMO - System B Query Classification")
    print("="*70)
    
    demo_queries = [
        # TRANSACTION_RISK
        "Given the transaction details, assess whether this transaction poses elevated risk.",
        "This doesn't look normal compared to past behavior — should I worry?",
        
        # MIXED_INTENT
        "Assess the risk and tell me if similar cases increased after demonetization.",
        "Evaluate this transaction and also explain how ML models detect fraud.",
        
        # GOVERNANCE
        "Explain the transaction risk and comment on RBI fraud prevention guidelines.",
        "Does this transaction comply with regulatory requirements?",
        
        # EXPLICIT_REFUSAL
        "Predict the market trend for this category.",
        "How do machine learning models typically detect fraud?",
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print(f"{'='*70}")
        
        route_info = router.classify_query(query)
        
        print(f" Route: {route_info['route']}")
        print(f" Confidence: {route_info['metadata']['confidence']}")
        print(f" Reason: {route_info['metadata']['reason']}")
        
        if route_info['should_refuse']:
            print(f"\n EXPLICIT REFUSAL")
            print(f"   System B will NOT call Finance SLM")
            print(f"   Response: \"{router.get_refusal_message(route_info)}\"")
        else:
            handling = router.get_handling_instructions(route_info)
            print(f"\n PROCESS MODE: {handling['mode'].upper()}")
            print(f"   System B will call Finance SLM")
            if handling['instruction']:
                print(f"   Special instruction: {handling['instruction'][:100]}...")
    
    print("\n" + "="*70)
    print(" Router Stats:", router.get_stats())
    print("="*70)


if __name__ == "__main__":
    demo()