import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Load the uploaded .npy files
train_features = np.load("train_features_combined.npy")
test_features = np.load("test_features_combined.npy")
train_labels = np.load("train_labels_combined.npy")
test_labels = np.load("test_labels_combined.npy")

# Combine train and test sets
features = np.vstack((train_features, test_features))
labels = np.concatenate((train_labels, test_labels))

# Reduce dimensions using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
features_2d = tsne.fit_transform(features)

# Plot the result
plt.figure(figsize=(10, 7))
sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels, palette="tab10", legend="full")
plt.title("t-SNE Visualization of Audio Features")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Emotion Label", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
