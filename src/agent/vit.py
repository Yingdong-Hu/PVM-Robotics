#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import timm.models.vision_transformer

from functools import partial
from collections import OrderedDict

from iopath.common.file_io import PathManagerFactory
pathmgr = PathManagerFactory.get()
from torchvision.transforms import Normalize


VIT_MODELS = {
    "mocov3-vit-b16": "vit-b-300ep.pth.tar",
    "dino-vit-b16": "dino_vitbase16_pretrain.pth",
    "mae-vit-b16": "mae_pretrain_vit_base.pth",
    "ibot-vit-b16": "ibot-vit-b16_checkpoint_teacher.pth",
    "clip-vit-b16": "CLIP-ViT-B-16.pt",
}


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer
        referene:
            - MAE:  https://github.com/facebookresearch/mae/blob/main/models_vit.py
            - timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, global_pool=False, encode_stackframes=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.encode_stackframes = encode_stackframes

        self.img_norm = Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                  std=torch.tensor([0.229, 0.224, 0.225]))
        del self.head

    def extract_feat(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)   # global pool without cls token
            # need layer norm
        else:
            x = self.norm(x)
            x = x[:, 0]

        return x

    def forward(self, x):
        x = x / 255.0

        if self.encode_stackframes:
            num_frame = x.shape[1] // 3
            feats = []
            for i in range(num_frame):
                img = x[:, i*3: (i+1)*3, ...]
                img = self.img_norm(img)
                feat = self.extract_feat(img)
                feats.append(feat)
            feats = torch.cat(feats, dim=-1)
            return feats
        else:
            img = x[:, -3:, ...]
            img = self.img_norm(img)
            feat = self.extract_feat(img)
            return feat

    @torch.no_grad()
    def infer_dims(self):
        in_sz = 224
        dummy = torch.zeros(1, 9, in_sz, in_sz).to(next(self.patch_embed.parameters()).device)
        dummy_out = self.forward(dummy)
        self.enc_hid_dim = dummy_out.shape[1]

    def freeze(self):
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False

        def _freeze_module(m):
            for p in m.parameters():
                p.requires_grad = False

        _freeze_module(self.patch_embed)
        _freeze_module(self.blocks)
        _freeze_module(self.norm)

        trainable_params = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_params.append(name)

        print(f"Trainable parameters in the encoder: {trainable_params}")


########## Clip Vision Transformer ##########
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ClipVisionTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 encode_stackframes: bool):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.encode_stackframes = encode_stackframes
        self.img_norm = Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                  std=(0.26862954, 0.26130258, 0.27577711))

    def extract_feat(self, x: torch.Tensor):
        x = self.conv1(x)                          # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)                     # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

    def forward(self, x: torch.Tensor):
        x = x / 255.0   # normalize to 0-1

        if self.encode_stackframes:
            num_frame = x.shape[1] // 3
            feats = []
            for i in range(num_frame):
                img = x[:, i*3: (i+1)*3, ...]
                img = self.img_norm(img)
                feat = self.extract_feat(img)
                feats.append(feat)
            feats = torch.cat(feats, dim=-1)
            return feats
        else:
            img = x[:, -3:, ...]
            img = self.img_norm(img)
            feat = self.extract_feat(img)
            return feat

    @torch.no_grad()
    def infer_dims(self):
        in_sz = 224
        dummy = torch.zeros(1, 9, in_sz, in_sz).to(next(self.conv1.parameters()).device)
        dummy_out = self.forward(dummy)
        self.enc_hid_dim = dummy_out.shape[1]

    def freeze(self):
        self.positional_embedding.requires_grad = False
        self.class_embedding.requires_grad = False
        self.proj.requires_grad = False

        def _freeze_module(m):
            for p in m.parameters():
                p.requires_grad = False

        _freeze_module(self.conv1)
        _freeze_module(self.transformer)
        _freeze_module(self.ln_pre)
        _freeze_module(self.ln_post)

        trainable_params = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_params.append(name)

        print(f"Trainable parameters in the encoder: {trainable_params}")


def vit_base_patch16(pretrained, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # infer hidden dim
    model.infer_dims()
    hidden_dim = model.enc_hid_dim

    assert os.path.exists(pretrained) or pretrained in ["none"]
    if pretrained != "none":
        load_checkpoint(pretrained, model)
        print("Loaded encoder from: {}".format(pretrained))
    return model, hidden_dim


def vit_clip(pretrained, **kwargs):
    model = ClipVisionTransformer(input_resolution=224, patch_size=16, width=768,
                                  layers=12, heads=12, output_dim=512, **kwargs)

    # infer hidden dim
    model.infer_dims()
    hidden_dim = model.enc_hid_dim
    assert os.path.exists(pretrained)
    load_clip_checkpoint(pretrained, model)
    print("Loaded encoder from: {}".format(pretrained))
    return model, hidden_dim


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def load_checkpoint(checkpoint_file, model):
    """Loads a checkpoint selectively based on the input options."""
    assert pathmgr.exists(checkpoint_file), "Checkpoint '{}' not found".format(
        checkpoint_file
    )
    with pathmgr.open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    print("Load ckpt from %s" % checkpoint_file)

    # state_dict = checkpoint["model"]
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "target_encoder" in checkpoint:
        state_dict = checkpoint["target_encoder"]
    else:
        state_dict = checkpoint

    # rename moco-v3 pre-trained keys
    if checkpoint_file.split('/')[-1] in ['vit-b-300ep.pth.tar']:
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            del state_dict[k]

    if checkpoint_file.split('/')[-1] in ['slip_base_100ep.pt', 'clip_base_25ep.pt']:
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.visual.') and not k.startswith('module.visual.head'):
                # remove prefix
                state_dict[k[len("module.visual."):]] = state_dict[k]
            del state_dict[k]

    # interpolate position embedding
    interpolate_pos_embed(model, state_dict)

    r = model.load_state_dict(state_dict, strict=False)
    if r.unexpected_keys or r.missing_keys:
        print(f"Loading weights, unexpected keys: {r.unexpected_keys}")
        print(f"Loading weights, missing keys: {r.missing_keys}")
    else:
        print("All keys matched successfully!")


def load_clip_checkpoint(checkpoint_file, model):
    assert pathmgr.exists(checkpoint_file), "Checkpoint '{}' not found".format(
        checkpoint_file
    )
    assert checkpoint_file.split('/')[-1] == 'CLIP-ViT-B-16.pt'

    with pathmgr.open(checkpoint_file, "rb") as f:
        pretrained_model = torch.jit.load(f, map_location="cpu").eval()
    print("Load ckpt from %s" % checkpoint_file)

    state_dict = pretrained_model.state_dict()

    for k in list(state_dict.keys()):
        if not k.startswith('visual.'):
            del state_dict[k]

    for k in list(state_dict.keys()):
        if k.startswith('visual.'):
            # remove prefix
            state_dict[k[len('visual.'):]] = state_dict[k]
        del state_dict[k]

    r = model.load_state_dict(state_dict, strict=False)
    if r.unexpected_keys or r.missing_keys:
        print(f"Loading weights, unexpected keys: {r.unexpected_keys}")
        print(f"Loading weights, missing keys: {r.missing_keys}")
    else:
        print("All keys matched successfully!")


def get_vit(root_dir, embedding_name, img_size, encode_stackframes=False):

    pretrain_fname = VIT_MODELS[embedding_name]
    pretrain_path = os.path.join(root_dir, 'pretrained', pretrain_fname)

    if embedding_name in ['mocov3-vit-b16', 'mae-vit-b16', 'ibot-vit-b16', 'dino-vit-b16']:
        model, hidden_dim = vit_base_patch16(pretrain_path, img_size=img_size,
                                             encode_stackframes=encode_stackframes)
    elif embedding_name in ['clip-vit-b16']:
        model, hidden_dim = vit_clip(pretrain_path, encode_stackframes=encode_stackframes)
    else:
        raise NotImplementedError

    return model, hidden_dim