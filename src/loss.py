"""
Cashew Pest and Disease Diagnosis System
Phase 3: Custom Loss Functions Engine (TensorFlow / Keras)
Supports Categorical Crossentropy and Categorical Focal Loss for Class Imbalance
"""

import tensorflow as tf
from tensorflow import keras


class CategoricalFocalLoss(keras.losses.Loss):
    """
    Categorical Focal Loss for handling severe class imbalance in vision models.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Compatible with Keras class_weight and sample_weight.
    Returns per-element loss tensor of shape (batch_size, num_classes) so Keras
    compile_loss can apply sample_weight / class_weight elementwise before batch reduction.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = keras.losses.Reduction.AUTO,
        name: str = "categorical_focal_loss",
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        # Cast y_true to match y_pred dtype for mixed_float16 precision safety
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        # Clip predictions to prevent numerical instability (log(0))
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate per-element cross entropy: shape (batch_size, num_classes)
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate per-element focal weight: shape (batch_size, num_classes)
        focal_weight = self.alpha * tf.math.pow(1.0 - y_pred, self.gamma)
        
        # Return per-element focal loss tensor of shape (batch_size, num_classes).
        # Keras compile_loss automatically multiplies this by sample_weight/class_weight
        # of shape (batch_size, num_classes) and applies SUM_OVER_BATCH_SIZE reduction.
        return focal_weight * cross_entropy

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha
        })
        return config


def get_loss_function(loss_name: str = "categorical_crossentropy", gamma: float = 2.0, alpha: float = 0.25):
    """
    Loss function selector for Keras model compilation.
    """
    name_lower = loss_name.lower()
    if name_lower in ["categorical_crossentropy", "crossentropy", "ce"]:
        return keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    elif name_lower in ["focal_loss", "focal", "fl"]:
        return CategoricalFocalLoss(gamma=gamma, alpha=alpha)
    else:
        raise ValueError(f"Unsupported loss function name: '{loss_name}'. Options: ['categorical_crossentropy', 'focal_loss']")
