import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np
import os
import re
from glob import glob

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load actual CoT length data from the report
def load_actual_cot_lengths():
    with open('/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/data/processed/serial_cot_generation_report_20250724_212449.json', 'r') as f:
        report_data = json.load(f)
    
    cot_lengths = {}
    for result in report_data['results']:
        cot_type = result['cot_type']
        avg_length = result['avg_length']
        
        # Map cot_type to model name format
        if cot_type == 'tiny':
            cot_lengths['qwen_tiny_cot'] = avg_length
        elif cot_type == 'short':
            cot_lengths['qwen_short_cot'] = avg_length
        elif cot_type == 'long':
            cot_lengths['qwen_long_cot'] = avg_length
        elif cot_type == 'very_long':
            cot_lengths['qwen_very_long_cot'] = avg_length
        elif cot_type == 'ultra_long':
            cot_lengths['qwen_ultra_long_cot'] = avg_length
    
    # Add baseline (no CoT)
    cot_lengths['baseline'] = 0
    
    return cot_lengths

def load_evaluation_results():
    """Load all evaluation result files"""
    results = {}
    
    # Get all evaluation result files
    eval_files = glob('/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/evaluation/evaluation_*.json')
    
    for file_path in eval_files:
        filename = os.path.basename(file_path)
        print(f"Processing file: {filename}")
        
        # Extract model name from filename
        # Format: evaluation_{model}_cuda_{gpu}_{timestamp}_{suffix}.json
        # Note: model names can contain underscores (very_long, ultra_long)
        match = re.search(r'evaluation_(.+)_cuda_\d+_\d+_\d+_\d+\.json', filename)
        if match:
            model_name = match.group(1)
            print(f"  Extracted model name: {model_name}")
            
            # Skip the medium model as requested
            if model_name == 'medium':
                print(f"  Skipping medium model as requested")
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                if 'detailed_results' in data and data['detailed_results']:
                    results[model_name] = data['detailed_results']
                    print(f"Loaded {len(data['detailed_results'])} results for {model_name}")
                else:
                    print(f"No detailed_results found in {filename}")
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"  No regex match for filename: {filename}")
    
    return results

def calculate_accuracy(results):
    """Calculate accuracy for each model"""
    model_accuracy = {}
    
    for model_name, model_results in results.items():
        correct_count = sum(1 for result in model_results if result.get('is_correct', False))
        total_count = len(model_results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        model_accuracy[model_name] = {
            'accuracy': accuracy,
            'correct': correct_count,
            'total': total_count
        }
        
        print(f"{model_name}: {correct_count}/{total_count} = {accuracy:.3f}")
    
    return model_accuracy

def map_model_names(model_accuracy, cot_lengths):
    """Map short model names to full CoT model names"""
    name_mapping = {
        'baseline': 'baseline',
        'tiny': 'qwen_tiny_cot',
        'short': 'qwen_short_cot',
        'long': 'qwen_long_cot',
        'very_long': 'qwen_very_long_cot',
        'ultra_long': 'qwen_ultra_long_cot'
    }
    
    mapped_data = []
    for short_name, stats in model_accuracy.items():
        full_name = name_mapping.get(short_name, short_name)
        if full_name in cot_lengths:
            mapped_data.append({
                'model': short_name.replace('_', ' ').title(),
                'cot_length': cot_lengths[full_name],
                'accuracy': stats['accuracy'],
                'correct': stats['correct'],
                'total': stats['total']
            })
    
    return mapped_data

def create_visualizations(data):
    """Create CoT length vs performance visualizations"""
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data available for visualization")
        return
    
    # Sort by CoT length for better visualization
    df = df.sort_values('cot_length')
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: CoT Length vs Accuracy (line plot)
    ax1.plot(df['cot_length'], df['accuracy'], marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('CoT Length (characters)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('CoT Length vs Model Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add data labels
    for i, row in df.iterrows():
        ax1.annotate(f'{row["model"]}\n{row["accuracy"]:.3f}', 
                    (row['cot_length'], row['accuracy']),
                    textcoords="offset points", xytext=(0,15), ha='center',
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Plot 2: Bar chart of accuracy by model
    bars = ax2.bar(df['model'], df['accuracy'], color=sns.color_palette("husl", len(df)))
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, acc in zip(bars, df['accuracy']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Scatter plot with trend line
    ax3.scatter(df['cot_length'], df['accuracy'], s=100, alpha=0.7, color=sns.color_palette("husl", len(df)))
    
    # Add trend line
    z = np.polyfit(df['cot_length'], df['accuracy'], 1)
    p = np.poly1d(z)
    ax3.plot(df['cot_length'], p(df['cot_length']), "r--", alpha=0.8, linewidth=2)
    
    ax3.set_xlabel('CoT Length (characters)', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('CoT Length vs Accuracy (with trend)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Add model labels
    for i, row in df.iterrows():
        ax3.annotate(row['model'], (row['cot_length'], row['accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Plot 4: Success rate table
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['model'],
            f"{row['cot_length']:.1f}",
            f"{row['correct']}/{row['total']}",
            f"{row['accuracy']:.3f}"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Model', 'CoT Length', 'Correct/Total', 'Accuracy'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
    
    ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = '/mnt/cvda/cvda_avatar/1/textarena-selfplay-qwen/cot_performance_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Calculate correlation
    if len(df) > 2:
        correlation = np.corrcoef(df['cot_length'], df['accuracy'])[0, 1]
        print(f"\nCorrelation between CoT length and accuracy: {correlation:.3f}")
    
    return df

def main():
    print("Loading actual CoT length data...")
    cot_lengths = load_actual_cot_lengths()
    print("CoT lengths:", cot_lengths)
    
    print("\nLoading evaluation results...")
    results = load_evaluation_results()
    
    if not results:
        print("No evaluation results found!")
        return
    
    print("\nCalculating model accuracy...")
    model_accuracy = calculate_accuracy(results)
    
    print("\nMapping model names and creating analysis data...")
    analysis_data = map_model_names(model_accuracy, cot_lengths)
    
    if not analysis_data:
        print("No matching data found between CoT lengths and evaluation results!")
        return
    
    print("\nCreating visualizations...")
    df = create_visualizations(analysis_data)
    
    print("\nAnalysis complete!")
    print(f"Analyzed {len(analysis_data)} models")
    
    # Print summary
    print("\nSummary:")
    for item in analysis_data:
        print(f"  {item['model']}: CoT Length = {item['cot_length']:.1f}, Accuracy = {item['accuracy']:.3f}")

if __name__ == "__main__":
    main()
