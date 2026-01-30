"""
Evaluation & Analysis - MoE vs Router+SLM Comparison

Analyzes experimental results comparing:
- System A: DeepSeek V3 (MoE Generalist)  
- System B: Qwen3 Finance + Router v2

Author: Your Name
Date: January 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List

# ===== CONFIGURATION =====

RESULTS_FILE = r"D:\Financial Assistant for nomad shareholders\SLM, MOE in Finance\modal_results.jsonl"
OUTPUT_DIR = Path(r'D:\Financial Assistant for nomad shareholders\SLM, MOE in Finance\backend\evaluate\analysis_results')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# ===== HELPER FUNCTIONS =====

def load_results(filepath: str) -> List[Dict]:
    """Load experiment results from JSONL file"""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def safe_get(obj, *keys, default=None):
    """Safely navigate nested dictionary or get attribute from object"""
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key)
        elif hasattr(obj, key):
            obj = getattr(obj, key)
        else:
            return default
        if obj is None:
            return default
    return obj

def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)

# ===== MAIN ANALYSIS =====

def main():
    print("="*70)
    print("📊 EXPERIMENT RESULTS ANALYSIS")
    print("   MoE vs Router+SLM Comparison")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading: {RESULTS_FILE}")
    results = load_results(RESULTS_FILE)
    df = pd.DataFrame(results)
    print(f"✅ Loaded {len(results)} experiments")
    
    # Extract metrics
    print("\n🔍 Extracting metrics...")
    df = extract_metrics(df)
    
    # Run analyses
    analyze_latency(df)
    analyze_routing(df)
    analyze_tokens(df)
    analyze_categories(df)
    analyze_reliability(df)
    
    # Generate visualizations
    generate_visualizations(df)
    
    # Export results
    export_results(df)
    
    # Final summary
    print_final_summary(df)

# ===== METRIC EXTRACTION =====

def extract_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract key metrics from raw results"""
    
    # System A metrics
    df['sys_a_latency'] = df.apply(
        lambda r: safe_get(r, 'system_a', 'latency_ms'), axis=1)
    df['sys_a_tokens'] = df.apply(
        lambda r: safe_get(r, 'system_a', 'metadata', 'tokens_used', default=0), axis=1)
    df['sys_a_error'] = df.apply(
        lambda r: safe_get(r, 'system_a', 'error'), axis=1)
    df['sys_a_refusal'] = df.apply(
        lambda r: safe_get(r, 'system_a', 'refusal', default=False), axis=1)
    
    # System B metrics
    df['sys_b_latency'] = df.apply(
        lambda r: safe_get(r, 'system_b', 'latency_ms'), axis=1)
    df['sys_b_tokens'] = df.apply(
        lambda r: safe_get(r, 'system_b', 'metadata', 'tokens_generated', default=0), axis=1)
    df['sys_b_route'] = df.apply(
        lambda r: safe_get(r, 'system_b', 'route'), axis=1)
    df['sys_b_error'] = df.apply(
        lambda r: safe_get(r, 'system_b', 'error'), axis=1)
    df['sys_b_refusal'] = df.apply(
        lambda r: safe_get(r, 'system_b', 'refusal', default=False), axis=1)
    df['sys_b_handling'] = df.apply(
        lambda r: safe_get(r, 'system_b', 'metadata', 'handling_mode'), axis=1)
    
    return df

# ===== ANALYSIS FUNCTIONS =====

def analyze_latency(df: pd.DataFrame):
    """Analyze and compare latency between systems"""
    print_section("⚡ 1. LATENCY ANALYSIS")
    
    sys_a = df['sys_a_latency'].dropna()
    sys_b = df['sys_b_latency'].dropna()
    
    # Statistics table
    stats = pd.DataFrame({
        'System A (MoE)': [
            sys_a.mean(),
            sys_a.median(),
            sys_a.std(),
            sys_a.min(),
            sys_a.max(),
            sys_a.quantile(0.95)
        ],
        'System B (Router+SLM)': [
            sys_b.mean(),
            sys_b.median(),
            sys_b.std(),
            sys_b.min(),
            sys_b.max(),
            sys_b.quantile(0.95)
        ]
    }, index=['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'P95'])
    
    print("\nLatency (milliseconds):")
    print(stats.round(2))
    
    # Comparison
    if sys_a.mean() < sys_b.mean():
        speedup = sys_b.mean() / sys_a.mean()
        print(f"\n⚡ System A is {speedup:.2f}x faster than System B")
    else:
        speedup = sys_a.mean() / sys_b.mean()
        print(f"\n⚡ System B is {speedup:.2f}x faster than System A")

def analyze_routing(df: pd.DataFrame):
    """Analyze routing behavior of System B"""
    print_section("🔀 2. ROUTING BEHAVIOR (System B)")
    
    routes = df['sys_b_route'].dropna()
    
    if len(routes) == 0:
        print("\n⚠️  No routing data found!")
        return
    
    route_counts = routes.value_counts()
    
    print("\nRoute Distribution:")
    for route, count in route_counts.items():
        pct = (count / len(routes)) * 100
        print(f"  {route:20s}: {count:3d} ({pct:5.1f}%)")
    
    # Special handling calculation
    normal_route = route_counts.get('transaction_risk', 0)
    special = len(routes) - normal_route
    
    print(f"\nBehavioral Resolution:")
    print(f"  Normal Processing: {normal_route} ({normal_route/len(routes)*100:.1f}%)")
    print(f"  Special Handling:  {special} ({special/len(routes)*100:.1f}%)")

def analyze_tokens(df: pd.DataFrame):
    """Analyze token usage and efficiency"""
    print_section("🎯 3. TOKEN EFFICIENCY")
    
    sys_a_total = df['sys_a_tokens'].sum()
    sys_b_total = df['sys_b_tokens'].sum()
    
    print(f"\nTotal Tokens:")
    print(f"  System A: {sys_a_total:,} tokens")
    print(f"  System B: {sys_b_total:,} tokens")
    
    print(f"\nAverage per Query:")
    print(f"  System A: {sys_a_total/len(df):.1f} tokens")
    print(f"  System B: {sys_b_total/len(df):.1f} tokens")
    
    if sys_a_total > sys_b_total:
        efficiency = sys_a_total / sys_b_total
        savings = (1 - sys_b_total/sys_a_total) * 100
        print(f"\n💰 System B is {efficiency:.2f}x more efficient ({savings:.0f}% fewer tokens)")
    else:
        efficiency = sys_b_total / sys_a_total
        savings = (1 - sys_a_total/sys_b_total) * 100
        print(f"\n💰 System A is {efficiency:.2f}x more efficient ({savings:.0f}% fewer tokens)")

def analyze_categories(df: pd.DataFrame):
    """Analyze performance by category"""
    print_section("📁 4. PERFORMANCE BY CATEGORY")
    
    category_stats = []
    for cat in sorted(df['category'].dropna().unique(), key=str):
        cat_df = df[df['category'] == cat]
        
        # Get primary route for this category
        routes = cat_df['sys_b_route'].dropna()
        primary_route = routes.mode()[0] if len(routes) > 0 else 'N/A'
        
        category_stats.append({
            'Category': cat,
            'Count': len(cat_df),
            'Sys A Latency': cat_df['sys_a_latency'].mean(),
            'Sys B Latency': cat_df['sys_b_latency'].mean(),
            'Sys A Tokens': cat_df['sys_a_tokens'].mean(),
            'Sys B Tokens': cat_df['sys_b_tokens'].mean(),
            'Primary Route': primary_route
        })
    
    cat_table = pd.DataFrame(category_stats)
    print("\n" + cat_table.to_string(index=False))

def analyze_reliability(df: pd.DataFrame):
    """Analyze system reliability and errors"""
    print_section("✅ 5. RELIABILITY & ERROR ANALYSIS")
    
    sys_a_errors = df['sys_a_error'].notna().sum()
    sys_b_errors = df['sys_b_error'].notna().sum()
    
    sys_a_refusals = df['sys_a_refusal'].sum()
    sys_b_refusals = df['sys_b_refusal'].sum()
    
    print(f"\nErrors:")
    print(f"  System A: {sys_a_errors}/{len(df)} ({sys_a_errors/len(df)*100:.1f}%)")
    print(f"  System B: {sys_b_errors}/{len(df)} ({sys_b_errors/len(df)*100:.1f}%)")
    
    print(f"\nRefusals:")
    print(f"  System A: {sys_a_refusals}/{len(df)} ({sys_a_refusals/len(df)*100:.1f}%)")
    print(f"  System B: {sys_b_refusals}/{len(df)} ({sys_b_refusals/len(df)*100:.1f}%)")
    
    print(f"\nSuccess Rate:")
    print(f"  System A: {(1 - sys_a_errors/len(df))*100:.1f}%")
    print(f"  System B: {(1 - sys_b_errors/len(df))*100:.1f}%")

# ===== VISUALIZATION =====

def generate_visualizations(df: pd.DataFrame):
    """Generate all visualizations"""
    print_section("📊 GENERATING VISUALIZATIONS")
    
    plot_latency_comparison(df)
    plot_route_distribution(df)
    plot_comparative_dashboard(df)
    plot_category_analysis(df)

def plot_latency_comparison(df: pd.DataFrame):
    """Plot latency comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sys_a = df['sys_a_latency'].dropna()
    sys_b = df['sys_b_latency'].dropna()
    
    # Box plot
    bp = ax1.boxplot([sys_a, sys_b], 
                      tick_labels=['System A\n(MoE)', 'System B\n(Router+SLM)'],
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#2ecc71')
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('Latency Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.hist(sys_a, bins=30, alpha=0.6, label='System A (MoE)', color='#e74c3c')
    ax2.hist(sys_b, bins=30, alpha=0.6, label='System B (Router+SLM)', color='#2ecc71')
    ax2.set_xlabel('Latency (ms)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Latency Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'latency_comparison.png', bbox_inches='tight')
    print(f"  ✅ latency_comparison.png")
    plt.close()

def plot_route_distribution(df: pd.DataFrame):
    """Plot routing distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    routes = df['sys_b_route'].value_counts()
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    bars = ax.bar(range(len(routes)), routes.values, 
                  color=colors[:len(routes)], alpha=0.8)
    ax.set_xticks(range(len(routes)))
    ax.set_xticklabels(routes.index, rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('System B: Routing Behavior (150 queries)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentages
    for i, (route, count) in enumerate(routes.items()):
        pct = (count / df['sys_b_route'].count()) * 100
        ax.text(i, count + 2, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'route_distribution.png', bbox_inches='tight')
    print(f"  ✅ route_distribution.png")
    plt.close()

def plot_comparative_dashboard(df: pd.DataFrame):
    """Plot 4-panel comparative dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Average Latency
    ax = axes[0, 0]
    systems = ['System A\n(MoE)', 'System B\n(Router+SLM)']
    latencies = [df['sys_a_latency'].mean(), df['sys_b_latency'].mean()]
    bars = ax.bar(systems, latencies, color=['#e74c3c', '#2ecc71'], alpha=0.7)
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Average Latency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(latencies)*0.02,
                f'{val:.0f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Panel 2: Special Handling
    ax = axes[0, 1]
    routes = df['sys_b_route'].dropna()
    normal = (routes == 'transaction_risk').sum()
    special = len(routes) - normal
    labels = ['System A\n(No Routing)', 'System B\n(Router v2)']
    special_rates = [0, (special/len(routes))*100]
    bars = ax.bar(labels, special_rates, color=['#95a5a6', '#3498db'], alpha=0.7)
    ax.set_ylabel('Special Handling (%)', fontsize=11)
    ax.set_title('Behavioral Resolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, special_rates):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Panel 3: Token Efficiency
    ax = axes[1, 0]
    avg_tokens = [df['sys_a_tokens'].mean(), df['sys_b_tokens'].mean()]
    bars = ax.bar(['System A', 'System B'], avg_tokens, 
                  color=['#e74c3c', '#2ecc71'], alpha=0.7)
    ax.set_ylabel('Avg Tokens/Query', fontsize=11)
    ax.set_title('Token Efficiency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, avg_tokens):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(avg_tokens)*0.02,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Panel 4: Success Rate
    ax = axes[1, 1]
    sys_a_errors = df['sys_a_error'].notna().sum()
    sys_b_errors = df['sys_b_error'].notna().sum()
    success_rates = [
        ((len(df) - sys_a_errors) / len(df)) * 100,
        ((len(df) - sys_b_errors) / len(df)) * 100
    ]
    bars = ax.bar(['System A', 'System B'], success_rates,
                  color=['#e74c3c', '#2ecc71'], alpha=0.7)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Reliability', fontsize=12, fontweight='bold')
    ax.set_ylim([95, 101])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparative_summary.png', bbox_inches='tight')
    print(f"  ✅ comparative_summary.png")
    plt.close()

def plot_category_analysis(df: pd.DataFrame):
    """Plot performance by category"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    categories = sorted(df['category'].dropna().unique(), key=str)
    cat_sys_a_lat = [df[df['category']==c]['sys_a_latency'].mean() for c in categories]
    cat_sys_b_lat = [df[df['category']==c]['sys_b_latency'].mean() for c in categories]
    
    # Latency by category
    x = np.arange(len(categories))
    width = 0.35
    ax1.bar(x - width/2, cat_sys_a_lat, width, label='System A', color='#e74c3c', alpha=0.7)
    ax1.bar(x + width/2, cat_sys_b_lat, width, label='System B', color='#2ecc71', alpha=0.7)
    ax1.set_xlabel('Category', fontsize=11)
    ax1.set_ylabel('Latency (ms)', fontsize=11)
    ax1.set_title('Latency by Category', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Tokens by category
    cat_sys_a_tok = [df[df['category']==c]['sys_a_tokens'].mean() for c in categories]
    cat_sys_b_tok = [df[df['category']==c]['sys_b_tokens'].mean() for c in categories]
    ax2.bar(x - width/2, cat_sys_a_tok, width, label='System A', color='#e74c3c', alpha=0.7)
    ax2.bar(x + width/2, cat_sys_b_tok, width, label='System B', color='#2ecc71', alpha=0.7)
    ax2.set_xlabel('Category', fontsize=11)
    ax2.set_ylabel('Tokens', fontsize=11)
    ax2.set_title('Token Usage by Category', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'category_analysis.png', bbox_inches='tight')
    print(f"  ✅ category_analysis.png")
    plt.close()

# ===== EXPORT =====

def export_results(df: pd.DataFrame):
    """Export summary and detailed results"""
    print_section("💾 EXPORTING RESULTS")
    
    # Summary JSON
    routes = df['sys_b_route'].dropna()
    normal_route = (routes == 'transaction_risk').sum()
    special = len(routes) - normal_route
    
    summary = {
        "metadata": {
            "total_experiments": len(df),
            "timestamp": pd.Timestamp.now().isoformat()
        },
        "latency": {
            "system_a_mean_ms": float(df['sys_a_latency'].mean()),
            "system_b_mean_ms": float(df['sys_b_latency'].mean()),
            "system_a_median_ms": float(df['sys_a_latency'].median()),
            "system_b_median_ms": float(df['sys_b_latency'].median()),
            "system_a_p95_ms": float(df['sys_a_latency'].quantile(0.95)),
            "system_b_p95_ms": float(df['sys_b_latency'].quantile(0.95))
        },
        "routing": {
            "distribution": {k: int(v) for k, v in df['sys_b_route'].value_counts().items()},
            "normal_processing_pct": float(normal_route/len(routes)*100),
            "special_handling_pct": float(special/len(routes)*100)
        },
        "tokens": {
            "system_a_total": int(df['sys_a_tokens'].sum()),
            "system_b_total": int(df['sys_b_tokens'].sum()),
            "system_a_avg": float(df['sys_a_tokens'].mean()),
            "system_b_avg": float(df['sys_b_tokens'].mean()),
            "efficiency_ratio": float(df['sys_a_tokens'].sum() / df['sys_b_tokens'].sum())
        },
        "reliability": {
            "system_a_errors": int(df['sys_a_error'].notna().sum()),
            "system_b_errors": int(df['sys_b_error'].notna().sum()),
            "system_a_success_rate_pct": float((1 - df['sys_a_error'].notna().sum()/len(df))*100),
            "system_b_success_rate_pct": float((1 - df['sys_b_error'].notna().sum()/len(df))*100)
        }
    }
    
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ summary.json")
    
    # Detailed CSV
    export_cols = ['experiment_id', 'category', 'query', 
                   'sys_a_latency', 'sys_a_tokens', 
                   'sys_b_latency', 'sys_b_route', 'sys_b_tokens']
    df[export_cols].to_csv(OUTPUT_DIR / 'detailed_results.csv', index=False)
    print(f"  ✅ detailed_results.csv")

# ===== FINAL SUMMARY =====

def print_final_summary(df: pd.DataFrame):
    """Print final summary of findings"""
    print_section("✅ ANALYSIS COMPLETE!")
    
    # Calculate key metrics
    sys_a_total_tokens = df['sys_a_tokens'].sum()
    sys_b_total_tokens = df['sys_b_tokens'].sum()
    efficiency = sys_a_total_tokens / sys_b_total_tokens
    
    routes = df['sys_b_route'].dropna()
    normal = (routes == 'transaction_risk').sum()
    special = len(routes) - normal
    
    print(f"\n📊 Key Findings:")
    print(f"  • Experiments: {len(df)}")
    print(f"  • System A Latency: {df['sys_a_latency'].mean():.0f}ms avg")
    print(f"  • System B Latency: {df['sys_b_latency'].mean():.0f}ms avg")
    print(f"  • Token Efficiency: System B is {efficiency:.2f}x more efficient")
    print(f"  • Special Handling: {special/len(routes)*100:.1f}% of queries")
    print(f"  • Success Rate: {(1-df['sys_a_error'].notna().sum()/len(df))*100:.0f}% (both systems)")
    
    print(f"\n📁 Output Directory: {OUTPUT_DIR}")
    print(f"  • latency_comparison.png")
    print(f"  • route_distribution.png")
    print(f"  • comparative_summary.png")
    print(f"  • category_analysis.png")
    print(f"  • summary.json")
    print(f"  • detailed_results.csv")
    
    print("\n🎉 Ready for Medium article!")

# ===== ENTRY POINT =====

if __name__ == "__main__":
    main()