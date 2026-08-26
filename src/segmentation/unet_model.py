"""
Cashew Pest and Disease Diagnosis System
Phase C.2: Production U-Net Architecture & Segmentation Metrics
Framework: TensorFlow / Keras
"""

from typing import Tuple, List, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K


# ---------------------------------------------------------
# 1. Numerically Stable Losses & Segmentation Metrics
# ---------------------------------------------------------
def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Computes differentiable Dice Coefficient for binary segmentation."""
    y_true_f = K.flatten(K.cast(y_true, "float32"))
    y_pred_f = K.flatten(K.cast(y_pred, "float32"))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Computes Dice Loss: 1.0 - Dice Coefficient."""
    return 1.0 - dice_coef(y_true, y_pred, smooth=smooth)


def iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Computes Intersection over Union (Jaccard Index)."""
    y_true_f = K.flatten(K.cast(y_true, "float32"))
    y_pred_f = K.flatten(K.cast(y_pred, "float32"))
    intersection = K.sum(y_true_f * y_pred_f)
    total = K.sum(y_true_f) + K.sum(y_pred_f)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)


def bce_dice_loss(bce_weight: float = 0.5, dice_weight: float = 0.5):
    """Combined Binary Crossentropy + Dice Loss."""
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        bce_val = bce(y_true, y_pred)
        dice_val = dice_loss(y_true, y_pred)
        return bce_weight * bce_val + dice_weight * dice_val

    loss.__name__ = "bce_dice_loss"
    return loss


# ---------------------------------------------------------
# 2. Production U-Net Architecture
# ---------------------------------------------------------
def _conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    dropout_rate: float = 0.0,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Standard double convolution block with BatchNormalization and ReLU."""
    prefix = f"{name}_" if name else ""

    # Conv 1
    x = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        padding="same",
        kernel_initializer="he_normal",
        name=f"{prefix}conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{prefix}bn1")(x)
    x = layers.Activation("relu", name=f"{prefix}relu1")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name=f"{prefix}drop")(x)

    # Conv 2
    x = layers.Conv2D(
        filters,
        (kernel_size, kernel_size),
        padding="same",
        kernel_initializer="he_normal",
        name=f"{prefix}conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{prefix}bn2")(x)
    x = layers.Activation("relu", name=f"{prefix}relu2")(x)

    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
    encoder_filters: Tuple[int, ...] = (32, 64, 128, 256),
    bottleneck_filters: int = 512,
    dropout_rate: float = 0.2,
    model_name: str = "Cashew_UNet",
) -> tf.keras.Model:
    """
    Constructs a robust U-Net architecture for cashew lesion segmentation.
    
    Inputs:
        input_shape: (224, 224, 3) normalized RGB float32.
        num_classes: 1 for binary lesion segmentation (sigmoid output).
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    skip_connections = []
    x = inputs

    # --- ENCODER PATH ---
    for i, filters in enumerate(encoder_filters):
        x = _conv_block(x, filters=filters, name=f"enc_block_{i+1}")
        skip_connections.append(x)
        x = layers.MaxPooling2D((2, 2), name=f"pool_{i+1}")(x)

    # --- BOTTLENECK ---
    x = _conv_block(
        x,
        filters=bottleneck_filters,
        dropout_rate=dropout_rate,
        name="bottleneck",
    )

    # --- DECODER PATH ---
    for i, filters in enumerate(reversed(encoder_filters)):
        skip = skip_connections[-(i + 1)]
        # Upsampling via Conv2DTranspose
        x = layers.Conv2DTranspose(
            filters,
            (2, 2),
            strides=(2, 2),
            padding="same",
            kernel_initializer="he_normal",
            name=f"up_transpose_{i+1}",
        )(x)
        # Concatenate skip connection
        x = layers.Concatenate(axis=-1, name=f"concat_{i+1}")([x, skip])
        x = _conv_block(x, filters=filters, name=f"dec_block_{i+1}")

    # --- OUTPUT LAYER ---
    outputs = layers.Conv2D(
        num_classes,
        (1, 1),
        activation="sigmoid",
        dtype="float32",
        name="segmentation_output",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model
