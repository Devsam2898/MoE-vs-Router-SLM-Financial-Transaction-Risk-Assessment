"""
Query Template Engine for Finance AI Testing

Generates diverse query formulations across 9 categories
to test system behavior comprehensively.
"""

import random
from typing import Dict, List

class QueryTemplates:
    """Generate queries across different categories to test LLM systems"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
    
    # ===== CATEGORY 1: Multi-Factor Risk Reasoning =====
    def category_1_queries(self, transaction: Dict) -> List[str]:
        """Core finance workload: multi-factor reasoning"""
        return [
            f"Given the transaction details and the user's historical behavior, assess whether this transaction poses elevated risk and justify your conclusion using at least two independent factors.",
            
            f"The transaction amount is ${transaction['amount']:,}, but the merchant category is {transaction['merchant_category']}. Explain whether the transaction should still be flagged and why.",
            
            f"Analyze this {transaction['merchant_category']} transaction for risk. Consider the amount (${transaction['amount']:,}) relative to the user's average (${transaction['user_avg_amount']:,}) and the account age ({transaction['account_age_months']} months)."
        ]
    
    # ===== CATEGORY 2: Counterfactual & Sensitivity Analysis =====
    def category_2_queries(self, transaction: Dict) -> List[str]:
        """What-if scenarios and decision boundary testing"""
        return [
            f"Identify the minimum change in transaction attributes that would reduce the assessed risk level, and explain why.",
            
            f"If the same transaction occurred after 36 months of account history instead of {transaction['account_age_months']} months, how would your assessment change?",
            
            f"What if the transaction amount was ${transaction['user_avg_amount']:,} instead of ${transaction['amount']:,}? Would your risk assessment change significantly?"
        ]
    
    # ===== CATEGORY 3: Implicit Domain Boundary Detection =====
    def category_3_queries(self, transaction: Dict) -> List[str]:
        """Tests for out-of-scope handling and boundary discipline"""
        return [
            f"Explain the transaction risk and briefly comment on whether this aligns with RBI fraud prevention guidelines.",
            
            f"Assess the transaction risk and suggest whether the user's financial habits indicate poor money management.",
            
            f"Evaluate this {transaction['merchant_category']} purchase for fraud risk and recommend if we should report this to the Financial Crimes Enforcement Network."
        ]
    
    # ===== CATEGORY 4: Mixed-Intent with Embedded Traps =====
    def category_4_queries(self, transaction: Dict) -> List[str]:
        """Realistic production queries with tricky elements"""
        return [
            f"Explain why this transaction is risky and also tell me if similar cases increased after demonetization.",
            
            f"Evaluate the transaction risk and write a short explanation suitable for a non-technical customer.",
            
            f"Is this ${transaction['amount']:,} {transaction['merchant_category']} transaction suspicious? Also, what's the current global fraud rate in this category?"
        ]
    
    # ===== CATEGORY 5: Adversarial Rephrasing =====
    def category_5_queries(self, transaction: Dict) -> List[str]:
        """Same intent, different surface forms - tests variance"""
        return [
            f"This doesn't look normal compared to past behavior — should I worry?",
            
            f"Given the deviation from baseline expenditure, determine if escalation is warranted.",
            
            f"Hey, this transaction seems off. What's your take on the risk here?"
        ]
    
    # ===== CATEGORY 9: Stress Testing for Over-Generalization =====
    def category_9_queries(self, transaction: Dict) -> List[str]:
        """Tests for hallucination and scope creep"""
        return [
            f"This transaction looks risky — summarize the global rise in financial fraud over the last decade.",
            
            f"What are the top 3 fraud prevention strategies used by Indian banks, and does this transaction fit any fraud patterns?",
            
            f"Assess the risk and also explain how machine learning models typically detect fraud in real-time."
        ]
    
    def generate_all_queries(self, transaction: Dict) -> Dict[str, List[str]]:
        """Generate queries from all categories for a given transaction"""
        return {
            "category_1": self.category_1_queries(transaction),
            "category_2": self.category_2_queries(transaction),
            "category_3": self.category_3_queries(transaction),
            "category_4": self.category_4_queries(transaction),
            "category_5": self.category_5_queries(transaction),
            "category_9": self.category_9_queries(transaction)
        }
    
    def generate_query_variants(self, transaction: Dict, queries_per_txn: int = 3) -> List[Dict]:
        """
        Generate diverse query variants for a transaction
        Returns list of {query, category, transaction} dicts
        """
        all_queries = self.generate_all_queries(transaction)
        
        # Select queries from different categories
        selected = []
        categories = list(all_queries.keys())
        random.shuffle(categories)
        
        for i in range(min(queries_per_txn, len(categories))):
            category = categories[i]
            query = random.choice(all_queries[category])
            selected.append({
                "query": query,
                "category": category,
                "transaction": transaction
            })
        
        return selected


def main():
    """Demo: Generate sample queries"""
    import json
    
    # Sample transaction
    sample_txn = {
        "transaction_id": "TXN_0001",
        "amount": 125000,
        "user_avg_amount": 8200,
        "merchant_category": "Electronics",
        "country": "India",
        "account_age_months": 4,
        "expected_risk_level": "HIGH",
        "expected_reason": "Amount deviates significantly from historical behavior"
    }
    
    templates = QueryTemplates()
    
    print(" Sample Queries for Transaction:")
    print(json.dumps(sample_txn, indent=2))
    print("\n" + "="*60 + "\n")
    
    all_queries = templates.generate_all_queries(sample_txn)
    
    for category, queries in all_queries.items():
        print(f" {category.upper()}")
        for i, query in enumerate(queries, 1):
            print(f"   {i}. {query}")
        print()


if __name__ == "__main__":
    main()