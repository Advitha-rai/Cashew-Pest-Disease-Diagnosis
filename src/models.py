"""
Cashew Pest and Disease Diagnosis System
Vision Model Factory - Supporting 10 Architectures (Standard & Advanced Vision Transformers)
"""

import logging
import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm

from src.config import Config
from src.utils import count_parameters, get_model_size_mb, get_logger

logger = get_logger("ModelFactory")

class VisionModelFactory:
    """
    Factory class for instantiating pre-trained vision models for transfer learning.
    Supports 10 architectures:
      1. MobileNetV2
      2. ResNet50
      3. VGG16
      4. InceptionV3
      5. DenseNet121
      6. EfficientNetV2
      7. MobileNetV3
      8. ConvNeXt
      9. Swin Transformer
      10. DINOv2
    """

    @staticmethod
    def create_model(
        model_name: str,
        num_classes: int = len(Config.DEFAULT_CLASSES),
        pretrained: bool = True,
        freeze_backbone: bool = True
    ) -> nn.Module:
        """
        Instantiates the requested architecture, attaches a custom classifier head,
        and optionally freezes backbone layers for initial transfer learning.
        """
        model_name = model_name.lower().strip()
        logger.info(f"Instantiating Model: '{model_name}' (Classes={num_classes}, Pretrained={pretrained})")

        model = None
        in_features = None

        # ---------------------------------------------------------
        # 1. MobileNetV2
        # ---------------------------------------------------------
        if "mobilenet_v2" in model_name or "mobilenetv2" in model_name:
            weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            model = tv_models.mobilenet_v2(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )

        # ---------------------------------------------------------
        # 2. ResNet50
        # ---------------------------------------------------------
        elif "resnet50" in model_name or "resnet" in model_name:
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            model = tv_models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )

        # ---------------------------------------------------------
        # 3. VGG16
        # ---------------------------------------------------------
        elif "vgg16" in model_name or "vgg" in model_name:
            weights = tv_models.VGG16_BN_Weights.DEFAULT if pretrained else None
            model = tv_models.vgg16_bn(weights=weights)
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(in_features, num_classes)
            )

        # ---------------------------------------------------------
        # 4. InceptionV3
        # ---------------------------------------------------------
        elif "inception_v3" in model_name or "inception" in model_name:
            weights = tv_models.Inception_V3_Weights.DEFAULT if pretrained else None
            model = tv_models.inception_v3(weights=weights, aux_logits=False)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(in_features, num_classes)
            )

        # ---------------------------------------------------------
        # 5. DenseNet121
        # ---------------------------------------------------------
        elif "densenet121" in model_name or "densenet" in model_name:
            weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = tv_models.densenet121(weights=weights)
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )

        # ---------------------------------------------------------
        # 6. EfficientNetV2
        # ---------------------------------------------------------
        elif "efficientnet_v2" in model_name or "efficientnet" in model_name:
            weights = tv_models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            model = tv_models.efficientnet_v2_s(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes)
            )

        # ---------------------------------------------------------
        # 7. MobileNetV3
        # ---------------------------------------------------------
        elif "mobilenet_v3" in model_name or "mobilenetv3" in model_name:
            weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            model = tv_models.mobilenet_v3_large(weights=weights)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)

        # ---------------------------------------------------------
        # 8. ConvNeXt (Tiny)
        # ---------------------------------------------------------
        elif "convnext" in model_name:
            try:
                weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
                model = tv_models.convnext_tiny(weights=weights)
                in_features = model.classifier[2].in_features
                model.classifier[2] = nn.Linear(in_features, num_classes)
            except Exception:
                model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)

        # ---------------------------------------------------------
        # 9. Swin Transformer (Swin-T)
        # ---------------------------------------------------------
        elif "swin" in model_name:
            try:
                weights = tv_models.Swin_T_Weights.DEFAULT if pretrained else None
                model = tv_models.swin_t(weights=weights)
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, num_classes)
            except Exception:
                model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)

        # ---------------------------------------------------------
        # 10. DINOv2 (Vision Transformer Foundation Model)
        # ---------------------------------------------------------
        elif "dinov2" in model_name or "dino" in model_name:
            try:
                # Load DINOv2 ViT-Small/14 backbone via Torch Hub or timm
                backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                embed_dim = backbone.embed_dim
                class DINOv2ForClassification(nn.Module):
                    def __init__(self, bb, dim, num_cls):
                        super().__init__()
                        self.backbone = bb
                        self.head = nn.Sequential(
                            nn.LayerNorm(dim),
                            nn.Dropout(0.3),
                            nn.Linear(dim, num_cls)
                        )
                    def forward(self, x):
                        features = self.backbone(x)
                        return self.head(features)
                model = DINOv2ForClassification(backbone, embed_dim, num_classes)
            except Exception as e:
                logger.warning(f"Fallback to timm ViT for DINOv2 due to hub load note: {e}")
                model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=pretrained, num_classes=num_classes)

        else:
            raise ValueError(f"Unsupported model name '{model_name}'. Select 1-10 from Config.MODEL_MAP.")

        # ---------------------------------------------------------
        # Freeze Backbone if requested (Phase 1 Transfer Learning)
        # ---------------------------------------------------------
        if freeze_backbone:
            VisionModelFactory.set_backbone_trainable(model, trainable=False)

        param_stats = count_parameters(model)
        size_mb = get_model_size_mb(model)
        logger.info(f"Model Summary [{model_name}]: {param_stats['total_parameters']:,} Total Params | "
                    f"{param_stats['trainable_parameters']:,} Trainable | Size: {size_mb:.2f} MB")

        return model

    @staticmethod
    def set_backbone_trainable(model: nn.Module, trainable: bool = True) -> None:
        """Freezes or unfreezes backbone parameters, keeping final classifier head trainable."""
        # Unfreeze all parameters first if trainable=True
        if trainable:
            for p in model.parameters():
                p.requires_grad = True
            logger.info("[FINE-TUNING PHASE] Backbone unfrozen for end-to-end training.")
            return

        # Freeze everything
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze final classification head layers
        classifier_names = ['classifier', 'fc', 'head']
        unfrozen_count = 0
        for name, child in model.named_children():
            if any(cn in name for cn in classifier_names):
                for p in child.parameters():
                    p.requires_grad = True
                    unfrozen_count += p.numel()

        logger.info(f"[CLASSIFIER PHASE] Backbone frozen. {unfrozen_count:,} classifier parameters trainable.")
