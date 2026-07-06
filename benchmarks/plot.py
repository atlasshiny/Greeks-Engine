import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create the 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load CPU data (Required)
if os.path.exists('cpu_results.csv'):
    cpu_df = pd.read_csv('cpu_results.csv', skiprows=8)
    cpu_df['key'] = cpu_df['name'].apply(lambda name: name.split('/')[1])
    cpu_df['Source'] = 'CPU'
else:
    print("Error: cpu_results.csv not found. Exiting.")
    exit()

# Check for GPU data
gpu_exists = os.path.exists('gpu_results.csv')
if gpu_exists:
    gpu_df = pd.read_csv('gpu_results.csv', skiprows=8)
    gpu_df['key'] = gpu_df['name'].apply(lambda name: name.split('/')[1])
    gpu_df['Source'] = 'GPU'
else:
    print("Warning: gpu_results.csv not found. Skipping GPU-dependent plots.")

# Set global aesthetic theme
sns.set_theme(style="whitegrid")

# CPU Only Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=cpu_df, x='key', y='real_time')
plt.title('CPU Performance')
plt.savefig('plots/cpu_performance.png')
plt.close()

# GPU Dependent Plots
if gpu_exists:
    combined_df = pd.concat([cpu_df, gpu_df])

    # Performance Trend (Pointplot)
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=combined_df, x='name', y='real_time', hue='Source', 
                  dodge=True, markers=['o', 's'], linestyles=['-', '--'])
    plt.title('Performance Trend (Pointplot)')
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('plots/performance_pointplot.png')
    plt.close()

    # Speedup Ratio Plot
    merged_perf = pd.merge(cpu_df[['key', 'real_time']], gpu_df[['key', 'real_time']], 
                           on='key', suffixes=('_cpu', '_gpu'))
    merged_perf['speedup'] = merged_perf['real_time_cpu'] / merged_perf['real_time_gpu']
    plt.figure(figsize=(10, 6))
    sns.barplot(data=merged_perf, x='key', y='speedup', palette='viridis')
    plt.axhline(1, color='red', linestyle='--')
    plt.title('Speedup Ratio (CPU Time / GPU Time)')
    plt.savefig('plots/speedup_ratio.png')
    plt.close()

    # GPU Memcpy Time Plot
    if 'Memcpy_GPU_ms' in gpu_df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=gpu_df.dropna(subset=['Memcpy_GPU_ms']), x='key', y='Memcpy_GPU_ms', color='orange')
        plt.title('GPU Memcpy Time (ms)')
        plt.savefig('plots/gpu_memcpy.png')
        plt.close()

    # GPU Execution Time Breakdown
    gpu_plot_df = gpu_df.dropna(subset=['real_time', 'Memcpy_GPU_ms']).copy()
    if not gpu_plot_df.empty:
        gpu_plot_df['Memcpy_ns'] = gpu_plot_df['Memcpy_GPU_ms'] * 1e6
        gpu_plot_df['Compute_ns'] = (gpu_plot_df['real_time'] - gpu_plot_df['Memcpy_ns']).clip(lower=0)
        subset = gpu_plot_df[['key', 'Compute_ns', 'Memcpy_ns']].melt(id_vars='key')
        plt.figure(figsize=(10, 6))
        sns.barplot(data=subset, x='key', y='value', hue='variable')
        plt.title('GPU Execution Time Breakdown')
        plt.savefig('plots/gpu_breakdown_plot.png')
        plt.close()

# CDF Plot (Can handle CPU only or Combined)
plt.figure(figsize=(10, 6))
sources = ['CPU', 'GPU'] if gpu_exists else ['CPU']
for source in sources:
    data = (combined_df if gpu_exists else cpu_df)
    data = data[data['Source'] == source]['real_time'].dropna().sort_values()
    if len(data) > 0:
        cdf = np.linspace(0, 1, len(data))
        plt.plot(data, cdf, label=source, marker='.')
plt.xscale('log')
plt.title('CDF of Execution Times')
plt.legend()
plt.savefig('plots/cdf_plot.png')
plt.close()