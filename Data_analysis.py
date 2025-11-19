import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif

df = pd.read_excel(input('Please input the path to your excel file: ').strip())

print("Outcome distribution")
outcome_counts = df['outcome'].value_counts().sort_index()
print(outcome_counts)
print("\nPercentages:")
print((outcome_counts / len(df) * 100).round(2))

# Figure 1: Outcome distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='outcome')
plt.title('Outcome Distribution')
for i, count in enumerate(outcome_counts):
    plt.text(i, count + 0.02*max(outcome_counts), str(count), ha='center')
plt.tight_layout()
plt.show()

# Figure 2: Feature histograms
plt.figure(figsize=(12, 8))
feature_cols = [col for col in df.columns if col != 'outcome']
df[feature_cols].hist(bins=30)
plt.tight_layout()
plt.show()

# Figure 3: Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

print("\nFeature outcome relationship")
X = df[feature_cols]
y = df['outcome']

f_values, p_values = f_classif(X, y)
feature_scores = pd.DataFrame({
    'Feature': feature_cols,
    'F_Value': f_values,
    'P_Value': p_values
}).sort_values('F_Value', ascending=False)

print(feature_scores.head(10))

# Figure 4: Top features vs outcome
top_features = feature_scores.head(4)['Feature'].values
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, feature in enumerate(top_features):
    sns.boxplot(data=df, x='outcome', y=feature, ax=axes[i])
    axes[i].set_title(f'{feature} vs Outcome')

plt.tight_layout()
plt.show()