
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure target directory exists
os.makedirs("www/img", exist_ok=True)

# Data
ages = np.arange(12, 21)
true_rates = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.22, 0.28, 0.35, 0.42])
model_rates = np.array([0.02, 0.03, 0.06, 0.09, 0.14, 0.18, 0.27, 0.32, 0.40])
scores = [60, 70, 78, 82, 88, 90, 92, 93, 95]
labels = ['yes', 'no']
true_dist = [0.15, 0.85]
model_dist = [0.14, 0.86]
uniform_dist = [0.5, 0.5]

# DataFrame
df = pd.DataFrame({
    'Age': ages,
    'True Smoking Rate': true_rates,
    'Model Estimate': model_rates
})

# 1. Smoking by age: true vs. model
plt.figure(figsize=(6, 4), facecolor='none')
plt.plot(df['Age'], df['True Smoking Rate'], label='True Distribution', marker='o', color='white')
plt.plot(df['Age'], df['Model Estimate'], label='Model Prediction', marker='s', color='skyblue')
plt.xlabel('Age')
plt.ylabel('Probability of Having Smoked')
plt.title('Smoking by Age: True vs. Model')
plt.legend()
plt.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.xticks(color='white')
plt.yticks(color='white')
plt.gca().spines[:].set_color('white')
plt.gca().title.set_color('white')
plt.gca().xaxis.label.set_color('white')
plt.gca().yaxis.label.set_color('white')
plt.tight_layout()
plt.savefig('www/img/smoking_by_age_plot.png', transparent=True)
plt.close()

# 2. Accuracy scores by age
plt.figure(figsize=(6, 4), facecolor='none')
plt.bar(df['Age'], scores, color='skyblue', edgecolor='white')
plt.xlabel('Age')
plt.ylabel('Score (out of 100)')
plt.title('Model Accuracy by Age (Smoking Example)')
plt.xticks(color='white')
plt.yticks(color='white')
plt.gca().spines[:].set_color('white')
plt.gca().title.set_color('white')
plt.gca().xaxis.label.set_color('white')
plt.gca().yaxis.label.set_color('white')
plt.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('www/img/smoking_example_results.png', transparent=True)
plt.close()

# 3. Distribution at age 16: true, model, uniform
plt.figure(figsize=(4.5, 3.5), facecolor='none')
x = range(len(labels))
w = 0.25
plt.bar([i - w for i in x], true_dist, w, label='True', edgecolor='white', color='white', alpha=0.8)
plt.bar(x, model_dist, w, label='Model', edgecolor='white', color='skyblue')
plt.bar([i + w for i in x], uniform_dist, w, label='Uniform', edgecolor='white', color='gray')
plt.xticks(x, labels, color='white')
plt.yticks(color='white')
plt.ylabel('Probability')
plt.title('Distribution at Age 16')
plt.legend()
plt.gca().spines[:].set_color('white')
plt.gca().title.set_color('white')
plt.gca().xaxis.label.set_color('white')
plt.gca().yaxis.label.set_color('white')
plt.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('www/img/distribution_age16.png', transparent=True)
plt.close()
