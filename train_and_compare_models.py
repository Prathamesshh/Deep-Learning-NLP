import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import time

# Load preprocessed data
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

with open('hamlet.txt', 'r') as f:
    text = f.read()

total_words = len(tokenizer.word_index) + 1
max_sequence_len = 15

# Recreate sequences (same as in notebook)
sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        sequences.append(n_gram_sequence)

sequences = np.array([np.pad(x, (max_sequence_len - len(x), 0), mode='constant') for x in sequences])

x = sequences[:, :-1]
y = sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

split_idx = int(0.8 * len(x))
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")
print(f"Total words: {total_words}, Max sequence length: {max_sequence_len}\n")

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ============ LSTM MODEL ============
print("=" * 60)
print("TRAINING LSTM MODEL")
print("=" * 60)

lstm_model = Sequential()
lstm_model.add(Input(shape=(max_sequence_len - 1,)))
lstm_model.add(Embedding(total_words, 100))
lstm_model.add(LSTM(150, return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(100))
lstm_model.add(Dense(total_words, activation="softmax"))
lstm_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

print(f"LSTM Model Parameters: {lstm_model.count_params()}")

lstm_start_time = time.time()
lstm_history = lstm_model.fit(
    x_train, y_train,
    epochs=50,
    validation_data=(x_test, y_test),
    verbose=1,
    callbacks=[early_stopping]
)
lstm_train_time = time.time() - lstm_start_time

lstm_model.save('lstm_model.h5')
print(f"\nLSTM Training completed in {lstm_train_time:.2f} seconds")

# ============ GRU MODEL ============
print("\n" + "=" * 60)
print("TRAINING GRU MODEL")
print("=" * 60)

gru_model = Sequential()
gru_model.add(Input(shape=(max_sequence_len - 1,)))
gru_model.add(Embedding(total_words, 100))
gru_model.add(GRU(150, return_sequences=True))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(100))
gru_model.add(Dense(total_words, activation="softmax"))
gru_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

print(f"GRU Model Parameters: {gru_model.count_params()}")

gru_start_time = time.time()
gru_history = gru_model.fit(
    x_train, y_train,
    epochs=50,
    validation_data=(x_test, y_test),
    verbose=1,
    callbacks=[early_stopping]
)
gru_train_time = time.time() - gru_start_time

gru_model.save('gru_model.h5')
print(f"\nGRU Training completed in {gru_train_time:.2f} seconds")

# ============ EVALUATION ============
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Get predictions
lstm_pred_probs = lstm_model.predict(x_test, verbose=0)
lstm_pred_classes = np.argmax(lstm_pred_probs, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

gru_pred_probs = gru_model.predict(x_test, verbose=0)
gru_pred_classes = np.argmax(gru_pred_probs, axis=1)

# Evaluate on test set
lstm_eval = lstm_model.evaluate(x_test, y_test, verbose=0)
gru_eval = gru_model.evaluate(x_test, y_test, verbose=0)

# Calculate detailed metrics
lstm_accuracy = lstm_eval[1]
lstm_loss = lstm_eval[0]
gru_accuracy = gru_eval[1]
gru_loss = gru_eval[0]

lstm_precision = precision_score(y_test_classes, lstm_pred_classes, average='weighted', zero_division=0)
lstm_recall = recall_score(y_test_classes, lstm_pred_classes, average='weighted', zero_division=0)
lstm_f1 = f1_score(y_test_classes, lstm_pred_classes, average='weighted', zero_division=0)

gru_precision = precision_score(y_test_classes, gru_pred_classes, average='weighted', zero_division=0)
gru_recall = recall_score(y_test_classes, gru_pred_classes, average='weighted', zero_division=0)
gru_f1 = f1_score(y_test_classes, gru_pred_classes, average='weighted', zero_division=0)

# Create comparison table
comparison_data = {
    'Metric': [
        'Test Loss',
        'Test Accuracy',
        'Precision (weighted)',
        'Recall (weighted)',
        'F1-Score (weighted)',
        'Training Time (seconds)',
        'Epochs Trained',
        'Parameters'
    ],
    'LSTM': [
        f"{lstm_loss:.4f}",
        f"{lstm_accuracy:.4f}",
        f"{lstm_precision:.4f}",
        f"{lstm_recall:.4f}",
        f"{lstm_f1:.4f}",
        f"{lstm_train_time:.2f}",
        f"{len(lstm_history.history['loss'])}",
        f"{lstm_model.count_params()}"
    ],
    'GRU': [
        f"{gru_loss:.4f}",
        f"{gru_accuracy:.4f}",
        f"{gru_precision:.4f}",
        f"{gru_recall:.4f}",
        f"{gru_f1:.4f}",
        f"{gru_train_time:.2f}",
        f"{len(gru_history.history['loss'])}",
        f"{gru_model.count_params()}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Calculate improvements
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"GRU Accuracy vs LSTM: {(gru_accuracy - lstm_accuracy):.4f} {'(GRU Better)' if gru_accuracy > lstm_accuracy else '(LSTM Better)'}")
print(f"GRU Loss vs LSTM: {(gru_loss - lstm_loss):.4f} {'(LSTM Better - Lower)' if gru_loss > lstm_loss else '(GRU Better - Lower)'}")
print(f"GRU F1-Score vs LSTM: {(gru_f1 - lstm_f1):.4f} {'(GRU Better)' if gru_f1 > lstm_f1 else '(LSTM Better)'}")
print(f"Training Time - LSTM: {lstm_train_time:.2f}s, GRU: {gru_train_time:.2f}s")

# ============ VISUALIZATIONS ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss comparison
axes[0, 0].plot(lstm_history.history['loss'], label='LSTM Train', marker='o')
axes[0, 0].plot(lstm_history.history['val_loss'], label='LSTM Validation', marker='s')
axes[0, 0].plot(gru_history.history['loss'], label='GRU Train', marker='^')
axes[0, 0].plot(gru_history.history['val_loss'], label='GRU Validation', marker='d')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training & Validation Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy comparison
axes[0, 1].plot(lstm_history.history['accuracy'], label='LSTM Train', marker='o')
axes[0, 1].plot(lstm_history.history['val_accuracy'], label='LSTM Validation', marker='s')
axes[0, 1].plot(gru_history.history['accuracy'], label='GRU Train', marker='^')
axes[0, 1].plot(gru_history.history['val_accuracy'], label='GRU Validation', marker='d')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Training & Validation Accuracy Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Test metrics comparison
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
lstm_metrics = [lstm_accuracy, lstm_precision, lstm_recall, lstm_f1]
gru_metrics = [gru_accuracy, gru_precision, gru_recall, gru_f1]

x_pos = np.arange(len(metrics_names))
width = 0.35

axes[1, 0].bar(x_pos - width/2, lstm_metrics, width, label='LSTM', alpha=0.8)
axes[1, 0].bar(x_pos + width/2, gru_metrics, width, label='GRU', alpha=0.8)
axes[1, 0].set_xlabel('Metrics')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Test Set Metrics Comparison')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(metrics_names)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Training time & epochs comparison
categories = ['LSTM', 'GRU']
train_times = [lstm_train_time, gru_train_time]
epochs = [len(lstm_history.history['loss']), len(gru_history.history['loss'])]

ax2 = axes[1, 1]
x_pos_time = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x_pos_time - width/2, train_times, width, label='Training Time (s)', alpha=0.8)
ax2.set_ylabel('Training Time (seconds)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

ax2_twin = ax2.twinx()
bars2 = ax2_twin.bar(x_pos_time + width/2, epochs, width, label='Epochs', alpha=0.8, color='orange')
ax2_twin.set_ylabel('Epochs', color='tab:orange')
ax2_twin.tick_params(axis='y', labelcolor='tab:orange')

ax2.set_xlabel('Model')
ax2.set_title('Training Time & Epochs Comparison')
ax2.set_xticks(x_pos_time)
ax2.set_xticklabels(categories)
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Comparison plot saved as 'model_comparison.png'")

# Save results to JSON
results = {
    'lstm': {
        'test_loss': float(lstm_loss),
        'test_accuracy': float(lstm_accuracy),
        'precision': float(lstm_precision),
        'recall': float(lstm_recall),
        'f1_score': float(lstm_f1),
        'training_time': lstm_train_time,
        'epochs_trained': len(lstm_history.history['loss']),
        'parameters': int(lstm_model.count_params())
    },
    'gru': {
        'test_loss': float(gru_loss),
        'test_accuracy': float(gru_accuracy),
        'precision': float(gru_precision),
        'recall': float(gru_recall),
        'f1_score': float(gru_f1),
        'training_time': gru_train_time,
        'epochs_trained': len(gru_history.history['loss']),
        'parameters': int(gru_model.count_params())
    }
}

with open('model_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Results saved to 'model_comparison_results.json'")

plt.show()
