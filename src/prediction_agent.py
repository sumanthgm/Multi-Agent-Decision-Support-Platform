"""
Prediction Agent — paper Section II-B-4 (Eq. 13).

Transformer-based 3-class sequence classifier over {Normal, Warning, Fault}.
2 encoder blocks, model_dim=64, 4 attention heads, feed-forward dim=128,
dropout=0.2, global average pooling, dense softmax head. Trained with sparse
categorical cross-entropy, Adam(1e-3), early stopping, class weighting.

This IS the "strong Transformer baseline" the paper compares ASPIRE against —
its raw argmax decisions are Table 5's "Transformer" row. Once trained it is
frozen and only its softmax outputs are consumed downstream by the Decision Agent.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def transformer_encoder_block(x, model_dim=64, num_heads=4, ff_dim=128, dropout=0.2):
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=model_dim // num_heads)(x, x)
    attn_out = layers.Dropout(dropout)(attn_out)
    x1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
    ff = layers.Dense(ff_dim, activation="relu")(x1)
    ff = layers.Dense(model_dim)(ff)
    ff = layers.Dropout(dropout)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(x1 + ff)


def build_prediction_agent(window_len: int, n_sensors: int, model_dim: int = 64,
                            num_heads: int = 4, ff_dim: int = 128,
                            dropout: float = 0.2, num_blocks: int = 2) -> tf.keras.Model:
    inp = layers.Input(shape=(window_len, n_sensors))
    x = layers.Dense(model_dim)(inp)  # project sensors -> model_dim
    for _ in range(num_blocks):
        x = transformer_encoder_block(x, model_dim, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(3, activation="softmax")(x)  # [P_normal, P_warn, P_fault]
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(y: np.ndarray) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    return {int(c): float(total / (len(classes) * n)) for c, n in zip(classes, counts)}


def train_prediction_agent(model: tf.keras.Model, X_train, y_train, X_val, y_val,
                            batch_size: int = 64, max_epochs: int = 100, patience: int = 10):
    cw = compute_class_weights(y_train)
    es = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), class_weight=cw,
              batch_size=batch_size, epochs=max_epochs, callbacks=[es], verbose=2)
    return model


def predict_probabilities(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """Returns (N, 3) array of [P_normal, P_warn, P_fault]."""
    return model.predict(X, verbose=0)
