"""
Cashew Pest and Disease Diagnosis System
Phase 3: Modular Model Architecture Factory (TensorFlow / Keras)
Supports 8 Pretrained ImageNet Architectures with Fine-Tuning Capabilities
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional

from src.config import Config


def get_base_backbone(model_name_key: str, input_shape: Tuple[int, int, int] = (224, 224, 3)):
    """
    Instantiates the requested pretrained ImageNet backbone from tf.keras.applications.
    """
    if model_name_key == "mobilenet_v2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif model_name_key == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif model_name_key == "vgg16":
        base_model = tf.keras.applications.VGG16(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif model_name_key == "inception_v3":
        base_model = tf.keras.applications.InceptionV3(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif model_name_key == "densenet121":
        base_model = tf.keras.applications.DenseNet121(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif model_name_key == "efficientnet_v2_b0":
        base_model = tf.keras.applications.EfficientNetV2B0(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif model_name_key == "mobilenet_v3_large":
        base_model = tf.keras.applications.MobileNetV3Large(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif model_name_key == "convnext_tiny":
        base_model = tf.keras.applications.ConvNeXtTiny(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    else:
        raise ValueError(f"Unsupported model architecture key: '{model_name_key}'")

    return base_model


def build_keras_model(
    model_index: int,
    num_classes: int = 4,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    trainable_backbone: bool = False,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Builds a complete Keras classification model combining a pretrained backbone
    with a custom global pooling and classification head.
    
    Args:
        model_index (int): Model selection index (1 to 8).
        num_classes (int): Number of target pest & disease classes.
        input_shape (Tuple[int, int, int]): Input tensor shape (224, 224, 3).
        trainable_backbone (bool): If True, unfreezes backbone weights.
        dropout_rate (float): Regularization dropout probability.
        
    Returns:
        keras.Model: Keras functional model ready for compilation and training.
    """
    if model_index not in Config.MODEL_MAP:
        raise ValueError(f"Invalid model index {model_index}. Must be between 1 and {len(Config.MODEL_MAP)}.")
    
    folder_name, model_name_key = Config.MODEL_MAP[model_index]
    print(f"[MODEL FACTORY] Initializing {folder_name} (Key: {model_name_key}) | Input: {input_shape} | Classes: {num_classes}")

    base_model = get_base_backbone(model_name_key, input_shape=input_shape)
    base_model.trainable = trainable_backbone

    inputs = keras.Input(shape=input_shape, name="input_image")
    
    # Pass inputs through backbone
    if model_name_key == "mobilenet_v3_large":
        x = base_model(inputs, training=False)
    else:
        x = base_model(inputs, training=trainable_backbone)
    
    # Custom Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    
    # Ensure float32 dtype for the final softmax layer to support Mixed Precision float16 training safely
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=folder_name)
    return model


def unfreeze_model_backbone(
    model: keras.Model,
    unfreeze_layers: Optional[int] = None,
    model_name_key: Optional[str] = None
) -> keras.Model:
    """
    Unfreezes backbone layers for fine-tuning.
    For MobileNetV3Large, performs controlled fine-tuning:
      - Freezes all layers 0..146 (first 147 layers).
      - Fine-tunes only non-BatchNormalization layers in top 40 layers.
      - Keeps ALL BatchNormalization layers frozen (trainable=False).
    For other models, unfreezes top N layers or all backbone layers.
    """
    base_model = None

    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_model = layer
            break

    if base_model is None:
        base_model = model

    if model_name_key == "mobilenet_v3_large":

        total_backbone_layers = len(base_model.layers)

        if total_backbone_layers != 187:
            raise RuntimeError(
                f"MobileNetV3Large backbone layer-count mismatch: "
                f"expected 187, found {total_backbone_layers}"
            )

        for layer in base_model.layers:
            layer.trainable = False

        bn_layers = [
            layer
            for layer in base_model.layers
            if isinstance(layer, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization))
            or "BatchNormalization" in layer.__class__.__name__
            or "BatchNorm" in layer.__class__.__name__
        ]

        top_40_layers = base_model.layers[-40:]

        for layer in top_40_layers:
            is_bn = (
                isinstance(layer, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization))
                or "BatchNormalization" in layer.__class__.__name__
                or "BatchNorm" in layer.__class__.__name__
            )
            if is_bn:
                layer.trainable = False
            else:
                layer.trainable = True

        for layer in bn_layers:
            layer.trainable = False

        trainable_backbone_layers = sum(
            1 for layer in base_model.layers
            if layer.trainable
        )

        trainable_bn_layers = sum(
            1 for layer in base_model.layers
            if (isinstance(layer, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization))
                or "BatchNormalization" in layer.__class__.__name__
                or "BatchNorm" in layer.__class__.__name__)
            and layer.trainable
        )

        frozen_earlier_layers = sum(
            1 for layer in base_model.layers[:-40]
            if not layer.trainable
        )

        total_bn_layers = len(bn_layers)

        if total_bn_layers != 46:
            raise RuntimeError(
                f"Expected 46 BatchNormalization layers, "
                f"found {total_bn_layers}"
            )

        if trainable_backbone_layers != 32:
            raise RuntimeError(
                f"Expected 32 trainable backbone layers, "
                f"found {trainable_backbone_layers}"
            )

        if trainable_bn_layers != 0:
            raise RuntimeError(
                f"Expected 0 trainable BatchNormalization layers, "
                f"found {trainable_bn_layers}"
            )

        if frozen_earlier_layers != 147:
            raise RuntimeError(
                f"Expected 147 frozen earlier layers, "
                f"found {frozen_earlier_layers}"
            )

        print("\n============================================================")
        print("[FINE-TUNING] MobileNetV3Large controlled fine-tuning")
        print(f"Total backbone layers: {total_backbone_layers}")
        print(f"Total BatchNormalization layers: {total_bn_layers}")
        print(f"Trainable backbone layers: {trainable_backbone_layers}")
        print(f"Trainable BatchNormalization layers: {trainable_bn_layers}")
        print(f"Frozen earlier backbone layers: {frozen_earlier_layers}")
        print("============================================================\n")

        return model

    # Existing behavior for all OTHER architectures only.
    base_model.trainable = True

    if unfreeze_layers is not None and unfreeze_layers > 0:
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        print(f"[FINE-TUNING] Unfroze top {unfreeze_layers} layers of backbone.")
    else:
        print(
            "[FINE-TUNING] Unfroze all backbone layers "
            "for full fine-tuning."
        )

    return model
