"""
Router Rules - Simple, Explainable Query Classification

This router uses transparent keyword-based rules to classify
financial queries into appropriate routing categories.

Design Philosophy:
- Simplicity over accuracy (explainability matters in finance)
- Deterministic (same input = same output always)
- Auditable (can explain why any decision was made)
"""

from typing import Dict, List, Tuple


class RouteClassifier:
    """Rule-based query classifier for financial AI routing"""
    
    # Define routing categories (in priority order)
    EXPLICIT_REFUSAL = "explicit_refusal"
    GOVERNANCE = "governance"
    MIXED_INTENT = "mixed_intent"
    TRANSACTION_RISK = "transaction_risk"
    
    # A. Hard refusal keywords (never answer at all)
    HARD_REFUSAL_KEYWORDS = [
        # Market/trading predictions
        "predict", "forecast", "will increase", "will decrease",
        "stock price", "market trend",
        
        # Personal financial advice
        "should i invest", "budget recommendation", "income bracket"
    ]
    
    # B. Governance / regulatory questions (conservative scope)
    GOVERNANCE_KEYWORDS = [
        "rbi guidelines", "rbi fraud prevention", "sebi", 
        "regulation", "compliance requirement", "legal requirement",
        "federal", "enforcement", "regulatory framework",
        "financial crimes enforcement"
    ]
    
    # C. Mixed-intent markers (partial answer needed)
    MIXED_INTENT_MARKERS = [
        "and also", "also tell me", "along with", "in addition",
        "and tell me", "also", "and what"
    ]
    
    # D. Macro / temporal knowledge (partial refusal)
    MACRO_TEMPORAL_KEYWORDS = [
        "demonetization", "after covid", "last year", "last decade",
        "recent trends", "global fraud rate", "current statistics",
        "latest data", "over the last decade", "rise in", "increased after",
        "global rise", "summarize the global"
    ]
    
    # E. General ML/AI questions (not about this transaction)
    GENERAL_ML_KEYWORDS = [
        "how do models", "machine learning", "ai detect", 
        "algorithm works", "fraud detection methods",
        "typical strategies", "how models work", "ml models"
    ]
    
    # F. Ethical/personal judgment questions
    ETHICAL_KEYWORDS = [
        "money management", "financial habits", "poor spending",
        "poor money management"
    ]
    
    # Keywords for transaction risk (in-scope)
    TRANSACTION_RISK_KEYWORDS = [
        "risk", "suspicious", "flag", "fraud", "unusual", "abnormal",
        "assess", "evaluate", "analyze", "examine", "review",
        "transaction", "amount", "behavior", "pattern", "deviation",
        "escalat", "worry", "concern", "alert"
    ]
    
    def __init__(self):
        """Initialize classifier with compiled keyword patterns"""
        # Convert all to lowercase for case-insensitive matching
        self.hard_refusal = [kw.lower() for kw in self.HARD_REFUSAL_KEYWORDS]
        self.governance = [kw.lower() for kw in self.GOVERNANCE_KEYWORDS]
        self.mixed_markers = [kw.lower() for kw in self.MIXED_INTENT_MARKERS]
        self.macro_temporal = [kw.lower() for kw in self.MACRO_TEMPORAL_KEYWORDS]
        self.general_ml = [kw.lower() for kw in self.GENERAL_ML_KEYWORDS]
        self.ethical = [kw.lower() for kw in self.ETHICAL_KEYWORDS]
        self.transaction_patterns = [kw.lower() for kw in self.TRANSACTION_RISK_KEYWORDS]
    
    def classify(self, query: str) -> Tuple[str, Dict]:
        """
        Classify a query into a routing category
        
        Priority order (highest to lowest):
        1. EXPLICIT_REFUSAL - Hard nos (predictions, investments)
        2. GOVERNANCE - Regulatory questions (conservative handling)
        3. MIXED_INTENT - Partial answer + refuse components
        4. TRANSACTION_RISK - Core use case (default)
        
        Args:
            query: The user's query string
        
        Returns:
            (route, metadata) tuple where:
            - route: The routing category
            - metadata: Dict with classification details (for debugging)
        """
        query_lower = query.lower()
        
        # Priority 1: Explicit refusal (never answer)
        hard_refusal_matches = [kw for kw in self.hard_refusal if kw in query_lower]
        if hard_refusal_matches:
            return self.EXPLICIT_REFUSAL, {
                "matched_keywords": hard_refusal_matches,
                "confidence": "high",
                "reason": f"Contains hard refusal triggers: {', '.join(hard_refusal_matches[:2])}"
            }
        
        # Priority 2: Governance / regulatory (conservative scope)
        governance_matches = [kw for kw in self.governance if kw in query_lower]
        if governance_matches:
            return self.GOVERNANCE, {
                "matched_keywords": governance_matches,
                "confidence": "high",
                "reason": f"Regulatory/governance question: {', '.join(governance_matches[:2])}"
            }
        
        # Priority 3: Mixed intent detection
        # Check if query has BOTH transaction risk AND out-of-scope elements
        transaction_matches = [kw for kw in self.transaction_patterns if kw in query_lower]
        
        # Check for mixed-intent markers or macro/temporal/ML keywords
        mixed_markers_found = [kw for kw in self.mixed_markers if kw in query_lower]
        macro_matches = [kw for kw in self.macro_temporal if kw in query_lower]
        ml_matches = [kw for kw in self.general_ml if kw in query_lower]
        ethical_matches = [kw for kw in self.ethical if kw in query_lower]
        
        out_of_scope_matches = macro_matches + ml_matches + ethical_matches
        
        # If query has transaction terms AND (mixed markers OR out-of-scope terms)
        if transaction_matches and (mixed_markers_found or out_of_scope_matches):
            return self.MIXED_INTENT, {
                "matched_keywords": {
                    "transaction": transaction_matches[:2],
                    "out_of_scope": out_of_scope_matches[:2]
                },
                "confidence": "high",
                "reason": f"Mixed intent: transaction risk + out-of-scope elements"
            }
        
        # Check for standalone out-of-scope (no transaction context)
        if out_of_scope_matches and not transaction_matches:
            return self.EXPLICIT_REFUSAL, {
                "matched_keywords": out_of_scope_matches,
                "confidence": "medium",
                "reason": f"Pure out-of-scope question: {', '.join(out_of_scope_matches[:2])}"
            }
        
        # Priority 4: Default to transaction_risk (core use case)
        return self.TRANSACTION_RISK, {
            "matched_keywords": transaction_matches if transaction_matches else ["default"],
            "confidence": "high" if transaction_matches else "default",
            "reason": "Transaction risk analysis" if transaction_matches else "Default to transaction risk"
        }
    
    def explain_classification(self, query: str) -> str:
        """
        Provide human-readable explanation of classification
        
        Useful for debugging and article screenshots
        """
        route, metadata = self.classify(query)
        
        explanation = f"Route: {route}\n"
        explanation += f"Confidence: {metadata['confidence']}\n"
        explanation += f"Reason: {metadata['reason']}\n"
        
        if metadata['matched_keywords'] and metadata['matched_keywords'] != ['default']:
            explanation += f"Matched keywords: {', '.join(metadata['matched_keywords'][:5])}"
        
        return explanation


def test_classifier():
    """Test the classifier with sample queries"""
    classifier = RouteClassifier()
    
    # Test cases with new 4-route system
    test_queries = [
        # Should route to transaction_risk
        ("Assess the risk of this transaction", "transaction_risk"),
        ("This doesn't look normal compared to past behavior", "transaction_risk"),
        ("Given the deviation from baseline, determine if escalation is warranted", "transaction_risk"),
        
        # Should route to explicit_refusal (hard nos)
        ("Predict the stock market trend", "explicit_refusal"),
        ("Should I invest in this merchant category?", "explicit_refusal"),
        
        # Should route to governance (regulatory questions)
        ("What are RBI guidelines for fraud prevention?", "governance"),
        ("Does this comply with SEBI regulations?", "governance"),
        
        # Should route to mixed_intent (partial answer + refuse)
        ("Assess risk and tell me if similar cases increased after demonetization", "mixed_intent"),
        ("Evaluate the transaction and also explain how ML models detect fraud", "mixed_intent"),
        
        # Should route to explicit_refusal (pure out-of-scope, no transaction context)
        ("How do machine learning models detect fraud?", "explicit_refusal"),
        ("Summarize the global rise in financial fraud over the last decade", "explicit_refusal"),
    ]
    
    print("="*70)
    print(" ROUTER CLASSIFICATION TESTS")
    print("="*70)
    
    correct = 0
    for query, expected_route in test_queries:
        route, metadata = classifier.classify(query)
        is_correct = route == expected_route
        correct += is_correct
        
        status = "Correct!" if is_correct else "No.."
        print(f"\n{status} Query: \"{query[:60]}...\"" if len(query) > 60 else f"\n{status} Query: \"{query}\"")
        print(f"   Expected: {expected_route} | Got: {route}")
        if not is_correct:
            print(f"   Reason: {metadata['reason']}")
    
    print("\n" + "="*70)
    print(f"Accuracy: {correct}/{len(test_queries)} ({correct/len(test_queries)*100:.0f}%)")
    print("="*70)


if __name__ == "__main__":
    test_classifier()