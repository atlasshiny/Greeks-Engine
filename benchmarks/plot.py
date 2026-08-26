import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_model(name):
    if 'BSM' in name: return 'BSM'
    elif 'Binomial' in name: return 'BinomialTree'
    elif 'MonteCarlo' in name or 'MC' in name: return 'MonteCarlo'
    return 'Unknown'

def parse_calc_type(name):
    if 'Price' in name: return 'Price'
    elif any(greek in name for greek in ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'Greek']): return 'Greek'
    return 'Other'

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
    df['Model'] = df['name'].apply(parse_model)
    df['CalcType'] = df['name'].apply(parse_calc_type)
    df['N'] = df['name'].apply(lambda x: int(x.split('/')[1]) if len(x.split('/')) > 1 else 0)
    return df

# Load Data
cpu_df = load_and_tag_data('cpu_results.csv', 'CPU')
gpu_df = load_and_tag_data('gpu_results.csv', 'GPU')

dataframes = [df for df in [cpu_df, gpu_df] if not df.empty]
if not dataframes:
    print("No data found to plot.")
    exit()

all_df = pd.concat(dataframes, ignore_index=True)
sns.set_theme(style="whitegrid")

models = ['BSM', 'BinomialTree', 'MonteCarlo']

for model in models:
    model_df = all_df[all_df['Model'] == model].copy()
    if model_df.empty:
        continue
        
    model_dir = f'plots/{model}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Root Level Model Plot: Price vs Greeks Comparison
    if len(model_df['CalcType'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=model_df, x='N', y='real_time', hue='CalcType', errorbar=None)
        plt.title(f'{model}: Price vs Greek Calculation Real Time')
        plt.ylabel('Real Time (ns)')
        plt.xlabel('Problem Size (N)')
        plt.savefig(f'{model_dir}/price_vs_greeks.png')
        plt.close()

    # Sub-folder Plotting per Calculation Type
    for calc_type in ['Price', 'Greek']:
        calc_df = model_df[model_df['CalcType'] == calc_type].copy()
        if calc_df.empty:
            continue
            
        calc_df.sort_values('N', inplace=True)
        calc_dir = f'{model_dir}/{calc_type}'
        os.makedirs(calc_dir, exist_ok=True)
        
        # CPU-Only Performance
        cpu_data = calc_df[calc_df['Source'] == 'CPU']
        if not cpu_data.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=cpu_data, x='N', y='real_time', color='skyblue')
            plt.title(f'{model} ({calc_type}): CPU Performance')
            plt.ylabel('Real Time (ns)')
            plt.xlabel('Problem Size (N)')
            plt.savefig(f'{calc_dir}/cpu_performance.png')
            plt.close()

        # GPU-Only Performance
        gpu_data = calc_df[calc_df['Source'] == 'GPU']
        if not gpu_data.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=gpu_data, x='N', y='real_time', color='salmon')
            plt.title(f'{model} ({calc_type}): GPU Performance')
            plt.ylabel('Real Time (ns)')
            plt.xlabel('Problem Size (N)')
            plt.savefig(f'{calc_dir}/gpu_performance.png')
            plt.close()
            
        # CPU vs GPU Performance Barplot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=calc_df, x='N', y='real_time', hue='Source')
        plt.title(f'{model} ({calc_type}): CPU vs GPU Real Time (Bar)')
        plt.ylabel('Real Time (ns)')
        plt.xlabel('Problem Size (N)')
        plt.savefig(f'{calc_dir}/performance_barplot.png')
        plt.close()

        # Performance Comparison - Pointplot (Log Scale Trend)
        plt.figure(figsize=(10, 6))
        sns.pointplot(data=calc_df, x='N', y='real_time', hue='Source', markers=['o', 's'], linestyles=['-', '--'])
        plt.title(f'{model} ({calc_type}): CPU vs GPU Real Time (Trend)')
        plt.ylabel('Real Time (ns)')
        plt.xlabel('Problem Size (N)')
        plt.yscale('log') 
        plt.savefig(f'{calc_dir}/performance_pointplot.png')
        plt.close()
        
        # GPU Breakdown (Memcpy & Compute)
        if not gpu_data.empty and 'Memcpy_GPU_ms' in gpu_data.columns:
            # Memcpy Time
            plt.figure(figsize=(10, 6))
            sns.barplot(data=gpu_data.dropna(subset=['Memcpy_GPU_ms']), x='N', y='Memcpy_GPU_ms', color='orange')
            plt.title(f'{model} ({calc_type}): GPU Memcpy Time (ms)')
            plt.ylabel('Memcpy Time (ms)')
            plt.xlabel('Problem Size (N)')
            plt.savefig(f'{calc_dir}/gpu_memcpy.png')
            plt.close()
            
            # Compute vs Memcpy Breakdown
            breakdown_df = gpu_data.copy().dropna(subset=['real_time', 'Memcpy_GPU_ms'])
            breakdown_df['Memcpy_ns'] = breakdown_df['Memcpy_GPU_ms'] * 1e6
            breakdown_df['Compute_ns'] = (breakdown_df['real_time'] - breakdown_df['Memcpy_ns']).clip(lower=0)
            subset = breakdown_df[['N', 'Compute_ns', 'Memcpy_ns']].melt(id_vars='N')
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=subset, x='N', y='value', hue='variable')
            plt.title(f'{model} ({calc_type}): GPU Execution Breakdown')
            plt.ylabel('Time (ns)')
            plt.xlabel('Problem Size (N)')
            plt.savefig(f'{calc_dir}/gpu_breakdown_plot.png')
            plt.close()

        # Speedup Ratio Plot
        pivot = calc_df.pivot(index='N', columns='Source', values='real_time')
        if 'CPU' in pivot.columns and 'GPU' in pivot.columns:
            pivot['Speedup'] = pivot['CPU'] / pivot['GPU']
            plt.figure(figsize=(10, 6))
            sns.barplot(x=pivot.index, y=pivot['Speedup'], color='seagreen')
            plt.axhline(1, color='red', linestyle='--', label='1x (Break-even)') 
            plt.title(f'{model} ({calc_type}): GPU Speedup Ratio (CPU / GPU)')
            plt.ylabel('Speedup (x)')
            plt.xlabel('Problem Size (N)')
            plt.legend()
            plt.savefig(f'{calc_dir}/speedup_ratio.png')
            plt.close()

        # CDF Plot
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(data=calc_df, x='real_time', hue='Source')
        plt.title(f'{model} ({calc_type}): Latency Distribution (CDF)')
        plt.xlabel('Real Time (ns)')
        plt.ylabel('Proportion')
        plt.savefig(f'{calc_dir}/cdf_plot.png')
        plt.close()

print("All plots generated successfully.")