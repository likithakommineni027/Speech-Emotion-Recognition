import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load preprocessed data
X_train = np.load('train_features_combined.npy')
y_train = np.load('train_labels_combined.npy')
X_test = np.load('test_features_combined.npy')
y_test = np.load('test_labels_combined.npy')

num_classes = len(np.unique(y_train))

# Compute class weights
classes = np.unique(y_train)
class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, class_weights_array))
print("\n📊 Class Weights:", class_weights)

# Define model
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

x = Conv1D(16, 3, padding='same', activation='relu')(input_layer)
x = BatchNormalization()(x)

x = Conv1D(32, 3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)

x = Conv1D(64, 3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)

x = Conv1D(128, 3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)

x = MaxPooling1D(pool_size=2)(x)

x = Bidirectional(LSTM(128, dropout=0.3, return_sequences=False))(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train
print("\n🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=70,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=2
)

model.save("ser_model.h5")
print("✅ Model saved to 'ser_model.h5'")

# Evaluate
print("\n📈 Evaluating on test set...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(f"\n✅ Test Accuracy: {accuracy_score(y_test, y_pred_classes):.4f}")
print(classification_report(y_test, y_pred_classes))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
class_names = [str(i) for i in range(num_classes)]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('📊 Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Train vs Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('📈 Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# t-SNE Visualization
print("\n🔍 Extracting features for t-SNE visualization...")
feature_extractor = Model(inputs=model.input, outputs=model.layers[-3].output)
features = feature_extractor.predict(X_test)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=y_test, cmap='tab10', alpha=0.7)
plt.title('🌈 t-SNE Visualization of Test Features')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.colorbar(scatter, label='True Labels')
plt.grid(True)
plt.tight_layout()
plt.show()  