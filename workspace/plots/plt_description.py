import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure target directory exists
os.makedirs("www/img", exist_ok=True)

# Data
ages = np.arange(12, 21)
true_rates = np.array([
    0.028,  # 12
    0.032,  # 13
    0.069,  # 14
    0.075,  # 15
    0.110,  # 16
    0.160,  # 17
    0.200,  # 18
    0.220,  # 19
    0.240   # 20
])
model_rates = np.array([
    0.020,  # 12
    0.03,  # 13
    0.065,  # 14
    0.067,  # 15
    0.095,  # 16
    0.140,  # 17
    0.205,  # 18
    0.210,  # 19
    0.230   # 20
])

df = pd.DataFrame({
    'Age': ages,
    'True Smoking Rate': true_rates,
    'Model Estimate': model_rates
})

# 1. Smoking by age
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

# Common settings for age 16 plots
labels = ['yes', 'no']
x = range(len(labels))
w = 0.25
uniform_dist = [0.5, 0.5]

# 2. distribution_age16.png (aligned)
true_dist = [0.11, 0.89]
model_dist = [0.095, 0.905]

plt.figure(figsize=(4.5, 3.5), facecolor='none')
plt.bar([i - w for i in x], true_dist, w, label='True', edgecolor='white', color='white', alpha=0.8)
plt.bar(x, model_dist, w, label='Model', edgecolor='white', color='skyblue')
plt.bar([i + w for i in x], uniform_dist, w, label='Uniform', edgecolor='white', color='gray')
plt.xticks(x, labels, color='white')
plt.yticks(color='white')
plt.ylabel('Probability')
plt.title('Distribution at Age 16 (Model A)')
plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
plt.gca().spines[:].set_color('white')
plt.gca().title.set_color('white')
plt.gca().xaxis.label.set_color('white')
plt.gca().yaxis.label.set_color('white')
plt.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('www/img/distribution_age16.png', transparent=True)
plt.close()

# 3. distribution_age16_off.png (misaligned model)
model_dist_off = [0.42, 0.58]

plt.figure(figsize=(4.5, 3.5), facecolor='none')
plt.bar([i - w for i in x], true_dist, w, label='True', edgecolor='white', color='white', alpha=0.8)
plt.bar(x, model_dist_off, w, label='Model', edgecolor='white', color='skyblue')
plt.bar([i + w for i in x], uniform_dist, w, label='Uniform', edgecolor='white', color='gray')
plt.xticks(x, labels, color='white')
plt.yticks(color='white')
plt.ylabel('Probability')
plt.title('Distribution at Age 16 (Model B)')
plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
plt.gca().spines[:].set_color('white')
plt.gca().title.set_color('white')
plt.gca().xaxis.label.set_color('white')
plt.gca().yaxis.label.set_color('white')
plt.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('www/img/distribution_age16_off.png', transparent=True)
plt.close()
