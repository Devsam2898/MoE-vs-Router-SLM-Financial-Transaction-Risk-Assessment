"""
Synthetic Transaction Generator for Finance AI Testing

Generates realistic transaction records with varying risk levels
for testing MoE vs SLM systems.
"""

import random
import json
from typing import Dict, List
from datetime import datetime

# Merchant categories by typical spending patterns
MERCHANT_CATEGORIES = {
    "low_risk": ["Grocery", "Pharmacy", "Gas Station", "Utilities"],
    "medium_risk": ["Restaurant", "Clothing", "Entertainment", "Online Shopping"],
    "high_risk": ["Electronics", "Jewelry", "International Transfer", "Crypto Exchange"]
}

COUNTRIES = ["India", "USA", "UK", "Singapore", "UAE", "Australia", "Germany"]

class TransactionGenerator:
    """Generates synthetic financial transactions with realistic patterns"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.transaction_count = 0
    
    def generate_normal_transaction(self) -> Dict:
        """Generate a low-risk transaction close to user's typical behavior"""
        user_avg = random.randint(2000, 15000)
        amount = int(user_avg * random.uniform(0.7, 1.3))  # Within 30% of average
        category = random.choice(MERCHANT_CATEGORIES["low_risk"])
        account_age = random.randint(12, 60)  # Established account
        
        self.transaction_count += 1
        return {
            "transaction_id": f"TXN_{str(self.transaction_count).zfill(4)}",
            "amount": amount,
            "user_avg_amount": user_avg,
            "merchant_category": category,
            "country": random.choice(COUNTRIES),
            "account_age_months": account_age,
            "expected_risk_level": "LOW",
            "expected_reason": "Transaction consistent with user behavior"
        }
    
    def generate_edge_case_transaction(self) -> Dict:
        """Generate medium-risk transaction with borderline attributes"""
        user_avg = random.randint(3000, 10000)
        # Amount is 2-3x average (borderline suspicious)
        amount = int(user_avg * random.uniform(2.0, 3.0))
        category = random.choice(MERCHANT_CATEGORIES["medium_risk"])
        account_age = random.randint(6, 12)  # Moderately new account
        
        self.transaction_count += 1
        return {
            "transaction_id": f"TXN_{str(self.transaction_count).zfill(4)}",
            "amount": amount,
            "user_avg_amount": user_avg,
            "merchant_category": category,
            "country": random.choice(COUNTRIES),
            "account_age_months": account_age,
            "expected_risk_level": "MEDIUM",
            "expected_reason": "Amount moderately exceeds typical spending, requires review"
        }
    
    def generate_high_risk_transaction(self) -> Dict:
        """Generate high-risk transaction with multiple red flags"""
        user_avg = random.randint(2000, 8000)
        # Amount significantly exceeds average
        amount = int(user_avg * random.uniform(10.0, 20.0))
        category = random.choice(MERCHANT_CATEGORIES["high_risk"])
        account_age = random.randint(1, 6)  # New account
        
        self.transaction_count += 1
        return {
            "transaction_id": f"TXN_{str(self.transaction_count).zfill(4)}",
            "amount": amount,
            "user_avg_amount": user_avg,
            "merchant_category": category,
            "country": random.choice(COUNTRIES),
            "account_age_months": account_age,
            "expected_risk_level": "HIGH",
            "expected_reason": "Amount deviates significantly from historical behavior, high-risk category, new account"
        }
    
    def generate_dataset(self, n_transactions: int = 50) -> List[Dict]:
        """
        Generate balanced dataset with mix of risk levels
        
        Distribution:
        - 50% normal transactions (low risk)
        - 30% edge cases (medium risk)
        - 20% high risk transactions
        """
        transactions = []
        
        # Calculate counts
        n_normal = int(n_transactions * 0.5)
        n_edge = int(n_transactions * 0.3)
        n_high_risk = n_transactions - n_normal - n_edge
        
        # Generate transactions
        for _ in range(n_normal):
            transactions.append(self.generate_normal_transaction())
        
        for _ in range(n_edge):
            transactions.append(self.generate_edge_case_transaction())
        
        for _ in range(n_high_risk):
            transactions.append(self.generate_high_risk_transaction())
        
        # Shuffle to mix risk levels
        random.shuffle(transactions)
        
        return transactions


def main():
    """Generate and save synthetic transactions"""
    generator = TransactionGenerator(seed=42)
    transactions = generator.generate_dataset(n_transactions=50)
    
    # Save to JSONL (one transaction per line)
    output_file = "transactions.jsonl"
    with open(output_file, 'w') as f:
        for txn in transactions:
            f.write(json.dumps(txn) + '\n')
    
    print(f" Generated {len(transactions)} transactions")
    print(f" Saved to: {output_file}")
    
    # Print summary stats
    risk_counts = {}
    for txn in transactions:
        level = txn['expected_risk_level']
        risk_counts[level] = risk_counts.get(level, 0) + 1
    
    print("\n Distribution:")
    for level, count in sorted(risk_counts.items()):
        print(f"   {level}: {count} ({count/len(transactions)*100:.0f}%)")


if __name__ == "__main__":
    main()