import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- 1. Load the Boston Housing dataset and prepare DataFrames ---

# Load the data.
(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()

# Define Column Names for the Features
feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]

# Create DataFrame for training features
X_train_df_original = pd.DataFrame(x_train, columns=feature_names)

# --- Apply Scalers ---

# Standard Scaler
standard_scaler = StandardScaler()
X_train_standard_scaled = standard_scaler.fit_transform(X_train_df_original)
X_train_standard_df = pd.DataFrame(X_train_standard_scaled, columns=feature_names)

# Min-Max Scaler
min_max_scaler = MinMaxScaler()
X_train_minmax_scaled = min_max_scaler.fit_transform(X_train_df_original)
X_train_minmax_df = pd.DataFrame(X_train_minmax_scaled, columns=feature_names)


# --- 2. Visualization ---

# Select a few features for comparison. 'CRIM' is highly skewed, 'RM' is more symmetric.
features_to_plot = ['CRIM', 'RM', 'TAX']

# Set up the plot style
sns.set_style("whitegrid")
plt.figure(figsize=(18, 15))

# Loop through each feature to plot its distribution before and after scaling
for i, feature in enumerate(features_to_plot):
    # Original Data - Histogram and KDE
    plt.subplot(len(features_to_plot), 3, i * 3 + 1)
    sns.histplot(X_train_df_original[feature], kde=True, bins=30, color='skyblue')
    plt.title(f'Original {feature}\n(Mean: {X_train_df_original[feature].mean():.2f}, Std: {X_train_df_original[feature].std():.2f})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Standard Scaled Data - Histogram and KDE
    plt.subplot(len(features_to_plot), 3, i * 3 + 2)
    sns.histplot(X_train_standard_df[feature], kde=True, bins=30, color='lightcoral')
    plt.title(f'Standard Scaled {feature}\n(Mean: {X_train_standard_df[feature].mean():.2f}, Std: {X_train_standard_df[feature].std():.2f})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim(-4, 6) # Set consistent x-axis for easier comparison

    # Min-Max Scaled Data - Histogram and KDE
    plt.subplot(len(features_to_plot), 3, i * 3 + 3)
    sns.histplot(X_train_minmax_df[feature], kde=True, bins=30, color='lightgreen')
    plt.title(f'Min-Max Scaled {feature}\n(Min: {X_train_minmax_df[feature].min():.2f}, Max: {X_train_minmax_df[feature].max():.2f})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim(-0.1, 1.1) # Set consistent x-axis for [0,1] range

plt.tight_layout()
plt.suptitle('Distribution of Features Before and After Normalization', y=1.02, fontsize=18)
plt.show()


# --- Box Plots for overall comparison ---
plt.figure(figsize=(18, 6))

# Original Data Box Plot
plt.subplot(1, 3, 1)
sns.boxplot(data=X_train_df_original[features_to_plot], palette='viridis')
plt.title('Original Data (Selected Features)')
plt.ylabel('Value')
plt.ylim(X_train_df_original[features_to_plot].min().min() - 5, X_train_df_original[features_to_plot].max().max() + 5) # Auto adjust limits better

# Standard Scaled Data Box Plot
plt.subplot(1, 3, 2)
sns.boxplot(data=X_train_standard_df[features_to_plot], palette='viridis')
plt.title('Standard Scaled Data')
plt.ylabel('Scaled Value')
plt.ylim(-3, 10) # Set consistent y-axis for standard scaled data

# Min-Max Scaled Data Box Plot
plt.subplot(1, 3, 3)
sns.boxplot(data=X_train_minmax_df[features_to_plot], palette='viridis')
plt.title('Min-Max Scaled Data')
plt.ylabel('Scaled Value')
plt.ylim(-0.1, 1.1) # Set consistent y-axis for min-max scaled data

plt.tight_layout()
plt.suptitle('Box Plots of Selected Features Before and After Normalization', y=1.02, fontsize=18)
plt.show()


# --- Scatter Plot for relationship visualization ---
# Let's pick two features, e.g., 'RM' (Avg rooms) and 'LSTAT' (% lower status population)

plt.figure(figsize=(15, 6))

# Original Scatter Plot
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_train_df_original['RM'], y=X_train_df_original['LSTAT'], color='blue', alpha=0.6)
plt.title('Original Data: RM vs LSTAT')
plt.xlabel('RM (Avg Rooms per Dwelling)')
plt.ylabel('LSTAT (% Lower Status Population)')

# Standard Scaled Scatter Plot
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_train_standard_df['RM'], y=X_train_standard_df['LSTAT'], color='red', alpha=0.6)
plt.title('Standard Scaled Data: RM vs LSTAT')
plt.xlabel('RM (Standardized)')
plt.ylabel('LSTAT (Standardized)')
plt.xlim(-3, 3) # Consistent limits
plt.ylim(-3, 3) # Consistent limits

plt.tight_layout()
plt.suptitle('Scatter Plot of Two Features Before and After Standardization', y=1.02, fontsize=16)
plt.show()
