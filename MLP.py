from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

# Load and normalize
X_train = np.load('train_features_combined.npy')
y_train = np.load('train_labels_combined.npy')
X_test = np.load('test_features_combined.npy')
y_test = np.load('test_labels_combined.npy')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Parameter grid
param_grid = {
    'hidden_layer_sizes': [
        (256, 128),
        (256, 128, 64),
        (512, 256, 128),
        (512, 256, 128, 64)
    ],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001],
    'max_iter': [500]
}

# Model and search
mlp = MLPClassifier(solver='adam', random_state=42)
grid = GridSearchCV(mlp, param_grid, cv=3, verbose=2, n_jobs=-1, scoring='accuracy', return_train_score=True)

print("🔍 Grid search running...")
grid.fit(X_train, y_train)

# Best model
best_mlp = grid.best_estimator_
print(f"✅ Best params: {grid.best_params_}")
print(f"📈 Cross-val accuracy: {grid.best_score_:.2f}")

# Evaluate
y_pred = best_mlp.predict(X_test)
print(f"\n🧪 Test accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Plot training vs validation accuracy
results = grid.cv_results_
plt.figure(figsize=(10, 6))
for i, params in enumerate(results['params']):
    train_score = results['mean_train_score'][i]
    val_score = results['mean_test_score'][i]
    plt.scatter(i, train_score, label='Train' if i == 0 else "", color='blue')
    plt.scatter(i, val_score, label='Validation' if i == 0 else "", color='orange')
plt.title("Train vs Validation Accuracy for Grid Search Models")
plt.xlabel("Model Index")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="viridis", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
features_2d = tsne.fit_transform(np.vstack((X_train, X_test)))
all_labels = np.concatenate((y_train, y_test))

plt.figure(figsize=(10, 7))
sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=all_labels, palette="tab10", legend="full")
plt.title("t-SNE Visualization of Audio Features (Train + Test)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Emotion Label", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()
