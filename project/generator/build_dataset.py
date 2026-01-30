"""
Experiment Dataset Builder

Combines synthetic transactions with query templates
to create complete test cases for MoE vs SLM comparison.
"""

import json
from typing import List, Dict
from generator import TransactionGenerator
from queries import QueryTemplates


class DatasetBuilder:
    """Build complete experimental dataset"""
    
    def __init__(self, seed: int = 28):
        self.txn_generator = TransactionGenerator(seed=seed)
        self.query_templates = QueryTemplates(seed=seed)
    
    def build_experiments(
        self, 
        n_transactions: int = 50,
        queries_per_transaction: int = 3
    ) -> List[Dict]:
        """
        Generate complete experimental dataset
        
        Args:
            n_transactions: Number of transactions to generate
            queries_per_transaction: Query variants per transaction
        
        Returns:
            List of experiment records with query + transaction context
        """
        # Generate transactions
        print(f" Generating {n_transactions} transactions...")
        transactions = self.txn_generator.generate_dataset(n_transactions)
        
        # Generate query variants for each transaction
        print(f" Generating {queries_per_transaction} queries per transaction...")
        experiments = []
        
        for txn in transactions:
            query_variants = self.query_templates.generate_query_variants(
                txn, 
                queries_per_transaction
            )
            experiments.extend(query_variants)
        
        # Add experiment IDs
        for i, exp in enumerate(experiments, 1):
            exp['experiment_id'] = f"EXP_{str(i).zfill(4)}"
        
        return experiments
    
    def save_dataset(self, experiments: List[Dict], output_file: str = "experiments.jsonl"):
        """Save experiments to JSONL format"""
        with open(output_file, 'w') as f:
            for exp in experiments:
                f.write(json.dumps(exp) + '\n')
        
        print(f" Saved {len(experiments)} experiments to {output_file}")
    
    def generate_summary(self, experiments: List[Dict]) -> Dict:
        """Generate dataset statistics"""
        category_counts = {}
        risk_counts = {}
        
        for exp in experiments:
            # Count by category
            cat = exp['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Count by risk level
            risk = exp['transaction']['expected_risk_level']
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        return {
            "total_experiments": len(experiments),
            "total_transactions": len(set(exp['transaction']['transaction_id'] for exp in experiments)),
            "queries_per_transaction": len(experiments) / len(set(exp['transaction']['transaction_id'] for exp in experiments)),
            "category_distribution": category_counts,
            "risk_distribution": risk_counts
        }


def main():
    """Generate complete experimental dataset"""
    print("="*60)
    print(" Building Experimental Dataset")
    print("="*60)
    print()
    
    builder = DatasetBuilder(seed=42)
    
    # Build dataset
    experiments = builder.build_experiments(
        n_transactions=50,
        queries_per_transaction=3
    )
    
    # Save to file
    builder.save_dataset(experiments, "experiments.jsonl")
    
    # Print summary statistics
    summary = builder.generate_summary(experiments)
    
    print("\n" + "="*60)
    print(" Dataset Summary")
    print("="*60)
    print(f"Total Experiments: {summary['total_experiments']}")
    print(f"Unique Transactions: {summary['total_transactions']}")
    print(f"Queries per Transaction: {summary['queries_per_transaction']:.1f}")
    
    print("\n🔹 Query Category Distribution:")
    for cat, count in sorted(summary['category_distribution'].items()):
        pct = count / summary['total_experiments'] * 100
        print(f"   {cat}: {count} ({pct:.1f}%)")
    
    print("\n🔹 Risk Level Distribution:")
    for risk, count in sorted(summary['risk_distribution'].items()):
        pct = count / summary['total_experiments'] * 100
        print(f"   {risk}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*60)
    print(" Dataset generation complete!")
    print("="*60)
    
    # Show a sample experiment
    print("\n Sample Experiment:")
    print(json.dumps(experiments[0], indent=2))


if __name__ == "__main__":
    main()