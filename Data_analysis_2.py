import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt

df = pd.read_excel(input('Please input the path to your excel file: ').strip())

# Check for missing values
print("Missing values")
print(df.isnull().sum())

# Check multicolinearity
print("Feature correlations")
feature_cols = [col for col in df.columns if col != 'outcome']
corr_matrix = df[feature_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature-Feature Correlations')
plt.show()

# Find high correlating features
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr.append((
                corr_matrix.columns[i], 
                corr_matrix.columns[j], 
                corr_matrix.iloc[i, j]
            ))
print("Highly correlated features (|r| > 0.8):")
for pair in high_corr:
    print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")

# Plot two highest impact features
plt.figure(figsize=(10, 8))
top_2_features = ['a_pc', 'b_pc']
sns.scatterplot(data=df, x=top_2_features[0], y=top_2_features[1], hue='outcome', palette='deep', alpha=0.6)
plt.title(f'Class Separation: {top_2_features[0]} vs {top_2_features[1]}')
plt.show()

# Simple rules
rules = [
    ('a_pc > 0.025', df['a_pc'] > 0.025, 0),
    ('v_km_s > 4.5 and v_km_s < 11 and a_pc > 0.011', 
     (df['v_km_s'] > 4.6) & (df['v_km_s'] < 10.5) & (df['a_pc'] > 0.011), 
     0) 
]

print("Simple rule based analysis")
total_coverage = 0
total_correct = 0

for rule_name, mask, predicted_outcome in rules:
    num_rule_applies = mask.sum()
    num_correct = (df[mask]['outcome'] == predicted_outcome).sum()
    accuracy = num_correct / num_rule_applies if num_rule_applies > 0 else 0
    
    print(f"Rule: {rule_name}, outcome {predicted_outcome}")
    print(f"Samples covered: {num_rule_applies} ({num_rule_applies/len(df)*100:.1f}%)")
    print(f"Correct: {num_correct} ({accuracy*100:.1f}% accuracy)")
    print(f"Solves: {num_correct/len(df)*100:.1f}% of total problem")
    
    total_coverage += num_rule_applies
    total_correct += num_correct

print(f"Summary: ")
print(f"Total coverage: {total_coverage} ({total_coverage/len(df)*100:.1f}% of data)")
print(f"Total correctly classified: {total_correct} ({total_correct/len(df)*100:.1f}% of total data)")
print(f"Remaining to classify: {len(df) - total_coverage} ({(len(df)-total_coverage)/len(df)*100:.1f}%)")


# Create 3 separate 2D plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: a_pc vs b_pc (top 2 features)
sns.scatterplot(data=df, x='a_pc', y='b_pc', hue='outcome', palette='deep', alpha=0.6, ax=axes[0])
axes[0].set_title('Top 2 Features: a_pc vs b_pc')
axes[0].set_xlabel('a_pc (Most Important)')
axes[0].set_ylabel('b_pc (2nd Most Important)')

# Plot 2: a_pc vs v_km_s (1st and 3rd features)
sns.scatterplot(data=df, x='a_pc', y='v_km_s', hue='outcome', palette='deep', alpha=0.6, ax=axes[1])
axes[1].set_title('a_pc vs v_km_s')
axes[1].set_xlabel('a_pc (Most Important)')
axes[1].set_ylabel('v_km_s (3rd Most Important)')

# Plot 3: b_pc vs v_km_s (2nd and 3rd features)
sns.scatterplot(data=df, x='b_pc', y='v_km_s', hue='outcome', palette='deep', alpha=0.6, ax=axes[2])
axes[2].set_title('b_pc vs v_km_s')
axes[2].set_xlabel('b_pc (2nd Most Important)')
axes[2].set_ylabel('v_km_s (3rd Most Important)')

plt.tight_layout()
plt.show()