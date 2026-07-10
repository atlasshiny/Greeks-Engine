import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration: Directories
os.makedirs('plots', exist_ok=True)

def load_and_tag_data(file_path, source_name):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return pd.DataFrame()
    
    header_row = 0
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('name,iterations'):
                header_row = i
                break
                
    df = pd.read_csv(file_path, skiprows=header_row)
    df['Source'] = source_name
    df['Model'] = df['name'].apply(lambda x: 'BSM' if 'BSM' in x else ('BinomialTree' if 'Binomial' in x else 'Unknown'))
    df['N'] = df['name'].apply(lambda x: int(x.split('/')[1]) if len(x.split('/')) > 1 else 0)
    return df

# 1. Load Data
cpu_df = load_and_tag_data('cpu_results.csv', 'CPU')
gpu_df = load_and_tag_data('gpu_results.csv', 'GPU')

dataframes = [df for df in [cpu_df, gpu_df] if not df.empty]
if not dataframes:
    print("No valid CSV data found. Exiting.")
    exit()

all_df = pd.concat(dataframes, ignore_index=True)
sns.set_theme(style="whitegrid")

# 2. Iterate by Model
for model in ['BSM', 'BinomialTree']:
    model_df = all_df[all_df['Model'] == model].copy()
    if model_df.empty:
        continue
        
    model_df.sort_values('N', inplace=True)
    base_dir = f'plots/{model}'
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. CPU-Only Performance (Added to generate cpu_performance.png)
    cpu_data = model_df[model_df['Source'] == 'CPU']
    if not cpu_data.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=cpu_data, x='N', y='real_time', color='skyblue')
        plt.title(f'{model}: CPU Performance')
        plt.ylabel('Real Time (ns)')
        plt.xlabel('Problem Size (N)')
        plt.savefig(f'{base_dir}/cpu_performance.png')
        plt.close()
    
    # 2. Performance Comparison - Barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=model_df, x='N', y='real_time', hue='Source')
    plt.title(f'{model}: CPU vs GPU Real Time (Bar)')
    plt.ylabel('Real Time (ns)')
    plt.xlabel('Problem Size (N)')
    plt.savefig(f'{base_dir}/performance_barplot.png')
    plt.close()

    # 3. Performance Comparison - Pointplot
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=model_df, x='N', y='real_time', hue='Source', markers=['o', 's'], linestyles=['-', '--'])
    plt.title(f'{model}: CPU vs GPU Real Time (Trend)')
    plt.ylabel('Real Time (ns)')
    plt.xlabel('Problem Size (N)')
    plt.yscale('log') 
    plt.savefig(f'{base_dir}/performance_pointplot.png')
    plt.close()
    
    # 4. GPU Breakdown
    gpu_data = model_df[model_df['Source'] == 'GPU']
    if not gpu_data.empty and 'Memcpy_GPU_ms' in gpu_data.columns:
        # Memcpy Time
        plt.figure(figsize=(10, 6))
        sns.barplot(data=gpu_data.dropna(subset=['Memcpy_GPU_ms']), x='N', y='Memcpy_GPU_ms', color='orange')
        plt.title(f'{model}: GPU Memcpy Time (ms)')
        plt.ylabel('Memcpy Time (ms)')
        plt.xlabel('Problem Size (N)')
        plt.savefig(f'{base_dir}/gpu_memcpy.png')
        plt.close()
        
        # Compute vs Memcpy Breakdown
        breakdown_df = gpu_data.copy().dropna(subset=['real_time', 'Memcpy_GPU_ms'])
        breakdown_df['Memcpy_ns'] = breakdown_df['Memcpy_GPU_ms'] * 1e6
        breakdown_df['Compute_ns'] = (breakdown_df['real_time'] - breakdown_df['Memcpy_ns']).clip(lower=0)
        subset = breakdown_df[['N', 'Compute_ns', 'Memcpy_ns']].melt(id_vars='N')
        plt.figure(figsize=(10, 6))
        sns.barplot(data=subset, x='N', y='value', hue='variable')
        plt.title(f'{model}: GPU Execution Breakdown')
        plt.ylabel('Time (ns)')
        plt.xlabel('Problem Size (N)')
        plt.savefig(f'{base_dir}/gpu_breakdown_plot.png')
        plt.close()

    # 5. Speedup Ratio
    pivot = model_df.pivot(index='N', columns='Source', values='real_time')
    if 'CPU' in pivot.columns and 'GPU' in pivot.columns:
        pivot['Speedup'] = pivot['CPU'] / pivot['GPU']
        plt.figure(figsize=(10, 6))
        sns.barplot(x=pivot.index, y=pivot['Speedup'], color='seagreen')
        plt.axhline(1, color='red', linestyle='--', label='1x (Break-even)') 
        plt.title(f'{model}: GPU Speedup Ratio (CPU / GPU)')
        plt.ylabel('Speedup (x)')
        plt.xlabel('Problem Size (N)')
        plt.legend()
        plt.savefig(f'{base_dir}/speedup_ratio.png')
        plt.close()

    # 6. CDF Plot
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=model_df, x='real_time', hue='Source')
    plt.title(f'{model}: Latency Distribution (CDF)')
    plt.xlabel('Real Time (ns)')
    plt.ylabel('Proportion')
    plt.savefig(f'{base_dir}/cdf_plot.png')
    plt.close()

print("All plots generated successfully including cpu_performance.png.")