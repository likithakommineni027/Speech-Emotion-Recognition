import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

# Load data
X_train = np.load('train_features_tess.npy')
X_test = np.load('test_features_tess.npy')
y_train = np.load('train_labels_tess.npy')
y_test = np.load('test_labels_tess.npy')

# Normalize features (apply BEFORE reshaping)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape to (samples, time_steps, features)
X_train = X_train.reshape(-1, 80, 62)
X_test = X_test.reshape(-1, 80, 62)

# Get number of emotion classes
num_classes = len(np.unique(y_train))

# Build Conv1D model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(80, 62)),
    BatchNormalization(),

    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),

    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),

    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
print("🚀 Training Conv1D model with normalization...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# Evaluate
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(f"\n✅ Test Accuracy: {accuracy_score(y_test, y_pred_classes):.2f}")
print(classification_report(y_test, y_pred_classes))