"""
Results Analysis & Visualization for Muon vs AdamW Experiment
Generates plots and statistical comparisons between optimizer performance
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/plunder/workspace/Knight2/knight')

class ExperimentAnalyzer:
    """Analyze results from Muon vs AdamW experiments"""
    
    def __init__(self, results_file: str):
        self.results = pd.read_csv(results_file)
        
    def summary_statistics(self):
        """Print summary of all experiment runs"""
        print("=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        for optimizer in self.results['optimizer'].unique():
            subset = self.results[self.results['optimizer'] == optimizer]
            
            avg_time = subset['training_time_s'].mean()
            max_accuracy = subset['evaluation_accuracy'].max()
            
            print(f"\n{optimizer.upper()}")
            print("-" * 40)
            print(f"  Runs: {len(subset)}")
            print(f"  Avg Training Time: {avg_time:.1f}s")
            print(f"  Max Accuracy: {max_accuracy:.2%}")
            
        return self.results
    
    def create_comparison_plots(self):
        """Generate comparison visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Training Time Comparison (Bar Chart)
        ax1 = axes[0, 0]
        avg_times = self.results.groupby('optimizer')['training_time_s'].mean()
        colors = ['#FF6B6B' if 'muon' in opt else '#4ECDC4' for opt in avg_times.index]
        bars = ax1.bar(avg_times.index, avg_times.values, color=colors, edgecolor='black')
        ax1.set_xlabel('Optimizer', fontsize=12)
        ax1.set_ylabel('Average Training Time (seconds)', fontsize=12)
        ax1.set_title('Training Efficiency Comparison', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_times.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.0f}s', ha='center', va='bottom', fontsize=11)
        
        # Plot 2: Accuracy Comparison (Bar Chart)
        ax2 = axes[0, 1]
        avg_accuracy = self.results.groupby('optimizer')['evaluation_accuracy'].mean()
        bars = ax2.bar(avg_accuracy.index, avg_accuracy.values * 100, color=colors, edgecolor='black')
        ax2.set_xlabel('Optimizer', fontsize=12)
        ax2.set_ylabel('Average Accuracy (%)', fontsize=12)
        ax2.set_title('Evaluation Performance Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_accuracy.values * 100):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
        
        # Plot 3: Time vs Accuracy Trade-off (Scatter)
        ax3 = axes[1, 0]
        for optimizer in self.results['optimizer'].unique():
            subset = self.results[self.results['optimizer'] == optimizer]
            ax3.scatter(subset['training_time_s'], 
                       subset['evaluation_accuracy'] * 100,
                       label=optimizer.upper(),
                       alpha=0.7, s=100)
        
        ax3.set_xlabel('Training Time (seconds)', fontsize=12)
        ax3.set_ylabel('Accuracy (%)', fontsize=12)
        ax3.set_title('Time vs Performance Trade-off', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Efficiency Ratio (Muon is faster = better)
        ax4 = axes[1, 1]
        efficiency_ratios = []
        for optimizer in self.results['optimizer'].unique():
            subset = self.results[self.results['optimizer'] == optimizer]
            
            # Calculate efficiency as accuracy / training_time
            if 'muon' in optimizer:
                baseline = self.results[self.results['optimizer'] != 'muon']['evaluation_accuracy'].mean()
                baseline_time = self.results[self.results['optimizer'] != 'muon']['training_time_s'].mean()
                
                muon_acc = subset['evaluation_accuracy'].mean()
                muon_time = subset['training_time_s'].mean()
                
                # Efficiency score: higher accuracy, lower time = better
                efficiency = (muon_acc / baseline) * (baseline_time / max(muon_time, 1))
            else:
                continue
            
            efficiency_ratios.append(efficiency)
        
        if efficiency_ratios:
            ax4.bar(['Muon Efficiency'], efficiency_ratios, color='#FF6B6B', edgecolor='black')
            ax4.axhline(y=1.0, linestyle='--', color='red', alpha=0.5, label='Baseline (AdamW)')
            ax4.set_ylabel('Efficiency Score (Accuracy × Time Savings)', fontsize=12)
            ax4.set_title('Overall Efficiency Comparison', fontsize=14, fontweight='bold')
            ax4.legend(fontsize=10)
        
        plt.tight_layout()
        output_path = os.path.join(Path(self.results_file).parent, 'results', 'comparison_plots.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {output_path}")
        
        return fig
    
    def statistical_comparison(self):
        """Perform basic statistical comparison between optimizers"""
        print("\n" + "=" * 60)
        print("STATISTICAL COMPARISON")
        print("=" * 60)
        
        if len(self.results['optimizer'].unique()) < 2:
            print("Not enough data for statistical comparison (need at least 2 optimizers)")
            return
        
        # Group by optimizer
        muon_data = self.results[self.results['optimizer'].str.contains('muon')]
        adamw_data = self.results[~self.results['optimizer'].str.contains('muon')]
        
        if len(muon_data) > 0 and len(adamw_data) > 0:
            # Calculate improvement metrics
            muon_time = muon_data['training_time_s'].mean()
            adamw_time = adamw_data['training_time_s'].mean()
            
            muon_acc = muon_data['evaluation_accuracy'].mean()
            adamw_acc = adamw_data['evaluation_accuracy'].mean()
            
            time_improvement = ((adamw_time - muon_time) / adamw_time) * 100
            accuracy_diff = (muon_acc - adamw_acc) * 100
            
            print(f"\nTime Efficiency:")
            print(f"  Muon avg: {muon_time:.1f}s")
            print(f"  AdamW avg: {adamw_time:.1f}s")
            print(f"  ** Time improvement: {time_improvement:+.1f}% **")
            
            print(f"\nAccuracy:")
            print(f"  Muon avg: {muon_acc:.2%}")
            print(f"  AdamW avg: {adamw_acc:.2%}")
            print(f"  Accuracy difference: {accuracy_diff:+.2f} percentage points")
            
            # Overall efficiency score
            muon_efficiency = muon_acc / max(muon_time, 1)
            adamw_efficiency = adamw_acc / max(adamw_time, 1)
            
            print(f"\nEfficiency Score (Accuracy/Time):")
            print(f"  Muon: {muon_efficiency:.6f}")
            print(f"  AdamW: {adamw_efficiency:.6f}")
            print(f"  ** Efficiency improvement: {(muon_efficiency/adamw_efficiency - 1) * 100:+.1f}% **")
        
        return muon_data, adamw_data

def main():
    """Main analysis function"""
    
    # Find results file
    base_dir = '/home/plunder/workspace/Knight2/knight/experiments/moon_vs_adamw'
    results_file = os.path.join(base_dir, 'results.csv')
    
    if not os.path.exists(results_file):
        print("No results found. Run experiments first with: python train.py")
        return
    
    # Initialize analyzer
    analyzer = ExperimentAnalyzer(results_file)
    
    # Print summary
    analyzer.summary_statistics()
    
    # Create visualizations
    fig = analyzer.create_comparison_plots()
    
    # Statistical comparison
    data = analyzer.statistical_comparison()
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check results directory for plots.")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Muon vs AdamW experiment results")
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    args = parser.parse_args()
    
    if args.plot:
        main()
    else:
        # Default: just print summary
        analyzer = ExperimentAnalyzer('/home/plunder/workspace/Knight2/knight/experiments/moon_vs_adamw/results.csv')
        analyzer.summary_statistics()

