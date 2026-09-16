import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def run_eda(input_csv='data/processed_transactions.csv', output_dir='visualizations'):
    if not os.path.exists(input_csv):
        input_csv = 'data/raw_transactions.csv'
        
    df = pd.read_csv(input_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style='darkgrid', palette='muted')
    
    # 1. Class Imbalance Analysis
    fraud_counts = df['is_fraud'].value_counts()
    fraud_ratio = df['is_fraud'].mean() * 100
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=['Normal (0)', 'Fraud (1)'], y=[fraud_counts.get(0, 0), fraud_counts.get(1, 0)], palette=['#2ecc71', '#e74c3c'])
    plt.title(f'Transaction Class Imbalance (Fraud Rate: {fraud_ratio:.2f}%)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Transactions')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_fraud_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Transaction Amount Distribution (Log Scale Boxplot)
    plt.figure(figsize=(10, 6))
    df['log_amount'] = np.log1p(df['amount'])
    sns.boxplot(x='is_fraud', y='log_amount', data=df, palette=['#2ecc71', '#e74c3c'])
    plt.xticks([0, 1], ['Normal', 'Fraud'])
    plt.title('Log Transaction Amount: Normal vs Fraud', fontsize=14, fontweight='bold')
    plt.xlabel('Transaction Type')
    plt.ylabel('Log(1 + Amount ($))')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_amount_distribution.png'), dpi=300)
    plt.close()
    
    # 3. Hourly Transaction Patterns
    hourly_df = df.groupby(['hour', 'is_fraud']).size().unstack(fill_value=0)
    hourly_df.columns = ['Normal', 'Fraud']
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    hourly_df.plot(kind='bar', stacked=True, ax=ax1, color=['#3498db', '#e74c3c'], alpha=0.85)
    fraud_rate_by_hour = (hourly_df['Fraud'] / (hourly_df['Normal'] + hourly_df['Fraud'])) * 100
    ax2.plot(fraud_rate_by_hour.index, fraud_rate_by_hour.values, color='#f1c40f', marker='o', linewidth=2.5, label='Fraud Rate %')
    
    ax1.set_title('Hourly Transaction Volume & Fraud Rate %', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Hour of Day (0-23)')
    ax1.set_ylabel('Transaction Count')
    ax2.set_ylabel('Fraud Rate (%)', color='#f1c40f')
    ax2.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_hourly_heatmap.png'), dpi=300)
    plt.close()
    
    # 4. Location Distance Anomaly Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df[df['is_fraud'] == 0]['distance_from_home_km'], label='Normal', color='#2ecc71', fill=True, alpha=0.4)
    sns.kdeplot(data=df[df['is_fraud'] == 1]['distance_from_home_km'], label='Fraud', color='#e74c3c', fill=True, alpha=0.4)
    plt.title('Distance From Home Location Density (km)', fontsize=14, fontweight='bold')
    plt.xlabel('Distance (km)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_distance_anomaly.png'), dpi=300)
    plt.close()
    
    print(f"[OK] Phase 3: EDA completed. Plots saved to '{output_dir}'")
    print(f"     Total Transactions: {len(df)}")
    print(f"     Normal: {fraud_counts.get(0, 0)} | Fraud: {fraud_counts.get(1, 0)} ({fraud_ratio:.2f}%)")
    print(f"     Avg Normal Amount: ${df[df['is_fraud']==0]['amount'].mean():.2f}")
    print(f"     Avg Fraud Amount:  ${df[df['is_fraud']==1]['amount'].mean():.2f}")

if __name__ == '__main__':
    run_eda()
