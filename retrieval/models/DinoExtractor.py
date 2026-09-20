import torch
import torch.nn as nn
from transformers import AutoModel


class DinoFeatureExtractor(nn.Module):
    """
    Unified extractor for HF DINOv2 / DINOv3-ViT.

    Args:
        ckpt_path: model path
        layers: intermediate layers to return, e.g. (3, 6, 9, 12)
        max_layer: truncate forward at this layer, e.g. 3 / 9 / 12 / None
                   None means full model
        freeze_until: freeze encoder layers [0, freeze_until)
                      e.g. 0 / 3 / 9 / 12
    """
    def __init__(self, ckpt_path, layers=None, max_layer=None, freeze_until=0, final_norm=False):
        super().__init__()
        self.layers = layers
        self.return_intermediate = False if layers is None else True

        self.patch_size = 16 if 'dinov3' in ckpt_path else 14
        self.backbone = AutoModel.from_pretrained(ckpt_path)

        config = self.backbone.config
        self.num_reg_tokens = getattr(config, "num_register_tokens", 0)

        self.max_layer = max_layer
        self.freeze_until = freeze_until
        self.final_norm = final_norm

        self._freeze_layers()

    def _split_tokens(self, x):
        cls_token = x[:, 0]  # [B, C]
        patch_tokens = x[:, 1 + self.num_reg_tokens:]  # [B, P, C]
        return cls_token, patch_tokens
    
    def _get_encoder_layers(self):
        # DINOv2
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return self.backbone.encoder.layer
        # DINOv3
        if hasattr(self.backbone, "layer"):
            return self.backbone.layer
    
    def _get_final_norm(self):
        if hasattr(self.backbone, "layernorm"):
            return self.backbone.layernorm
        if hasattr(self.backbone, "norm"):
            return self.backbone.norm
        return nn.Identity()

    def _freeze_layers(self):
        if self.freeze_until <= 0:
            return
        
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False

        # freeze first freeze_until encoder blocks
        encoder_layers = self._get_encoder_layers()
        for i, layer in enumerate(encoder_layers):
            if i < self.freeze_until:
                for p in layer.parameters():
                    p.requires_grad = False

    def _forward_truncated(self, x):
        hidden_states_dict = {}

        # embedding output
        h = self.backbone.embeddings(pixel_values=x)
        hidden_states_dict[0] = h

        layers = self._get_encoder_layers()
        total_layers = len(layers)
        max_layer = total_layers if self.max_layer is None else self.max_layer
        max_layer = min(max_layer, total_layers)

        position_embeddings = None
        if hasattr(self.backbone, "rope_embeddings"):
            position_embeddings = self.backbone.rope_embeddings(x)

        for i in range(max_layer):
            layer_out = layers[i](h, position_embeddings=position_embeddings) if position_embeddings is not None else layers[i](h)

            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            hidden_states_dict[i + 1] = h

        # final norm
        if self.final_norm:
            norm = self._get_final_norm()
            h = norm(h)

        return h, hidden_states_dict

    def forward(self, x):
        # full forward
        if self.max_layer is None:
            outputs = self.backbone(
                pixel_values=x,
                output_hidden_states=self.return_intermediate,
                return_dict=True,)

            final_cls, final_patch = self._split_tokens(outputs.last_hidden_state)

            out = {
                "final_cls": final_cls,
                "final_patch": final_patch,
                "intermediate": None,
            }

            if self.return_intermediate:
                inter = {}
                for i in self.layers:
                    h = outputs.hidden_states[i]
                    cls_i, patch_i = self._split_tokens(h)
                    inter[i] = {
                        "cls": cls_i,
                        "patch": patch_i,
                    }
                out["intermediate"] = inter

            return out

        # truncated forward
        final_hidden, hidden_states_dict = self._forward_truncated(x)
        final_cls, final_patch = self._split_tokens(final_hidden)

        out = {
            "final_cls": final_cls,
            "final_patch": final_patch,
            "intermediate": None,
        }

        if self.return_intermediate:
            inter = {}
            for i in self.layers:
                if i in hidden_states_dict:   # 只返回实际跑到的层
                    cls_i, patch_i = self._split_tokens(hidden_states_dict[i])
                    inter[i] = {
                        "cls": cls_i,
                        "patch": patch_i,
                    }
            out["intermediate"] = inter

        return out