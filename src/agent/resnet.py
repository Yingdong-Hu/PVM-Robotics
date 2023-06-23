import torch
import torch.nn as nn
import torch.nn.functional as F
import os

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torchvision.models.resnet as torch_resnet
from torchvision.models.resnet import BasicBlock, Bottleneck

from iopath.common.file_io import PathManagerFactory
pathmgr = PathManagerFactory.get()
from torchvision.transforms import Normalize


RESNET_MODELS = {
    "mocov2-resnet50": "moco_v2_800ep_pretrain.pth.tar",
    "swav-resnet50": "swav_800ep_pretrain.pth.tar",
    "simsiam-resnet50": "checkpoint_0099.pth.tar",
    "densecl-resnet50": "densecl_r50_imagenet_200ep.pth",
    "pixpro-resnet50": "pixpro_base_r50_400ep_md5_919c6612.pth",
    "vicregl-resnet50": "resnet50_alpha0.9.pth",
    "vfs-resnet50": "r50_nc_sgd_cos_100e_r5_1xNx2_k400-d7ce3ad0.pth",
    "r3m-resnet50": "original_r3m.pt",
    "vip-resnet50_VIPfc": "vip.pt",
}

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
              'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
              'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
              }


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class ResNet(torch_resnet.ResNet):
    def __init__(self, *args, **kwargs):
        self.encode_stackframes = kwargs.pop('encode_stackframes')
        super(ResNet, self).__init__(*args, **kwargs)

        del self.fc
        self.img_norm = Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                  std=torch.tensor([0.229, 0.224, 0.225]))

    def modify(self, remove_layers=[]):
        filter_layers = lambda x: [l for l in x if getattr(self, l) is not None]
        for layer in filter_layers(remove_layers):
            print("remove {}".format(layer))
            delattr(self, layer)

    def extract_feat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = x if not hasattr(self, 'layer3') else self.layer3(x)
        x = x if not hasattr(self, 'layer4') else self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if hasattr(self, 'fc'):
            x = self.fc(x)
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
        dummy = torch.zeros(1, 9, in_sz, in_sz).to(next(self.layer1.parameters()).device)
        dummy_out = self.forward(dummy)
        self.enc_hid_dim = dummy_out.shape[1]

    def freeze(self):

        def _freeze_module(m):
            for p in m.parameters():
                p.requires_grad = False

        _freeze_module(self.conv1)
        _freeze_module(self.bn1)
        _freeze_module(self.layer1)
        _freeze_module(self.layer2)
        if hasattr(self, 'layer3'):
            _freeze_module(self.layer3)
        if hasattr(self, 'layer4'):
            _freeze_module(self.layer4)
        if hasattr(self, 'fc'):
            _freeze_module(self.fc)

        trainable_params = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                trainable_params.append(name)

        print(f"Trainable parameters in the encoder: {trainable_params}")


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        msg = model.load_state_dict(state_dict, strict=False)
        if msg.unexpected_keys or msg.missing_keys:
            print(f"Loading weights, unexpected keys: {msg.unexpected_keys}")
            print(f"Loading weights, missing keys: {msg.missing_keys}")
        else:
            print("All keys matched successfully!")
    return model


def resnet50(pretrained=False, progress=True, **kwargs) -> ResNet:
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def unwrap_model(model):
    """Remove the DistributedDataParallel wrapper if present."""
    wrapped = isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)
    return model.module if wrapped else model


def load_checkpoint(checkpoint_file, model):
    """Loads a checkpoint selectively based on the input options."""
    assert pathmgr.exists(checkpoint_file), "Checkpoint '{}' not found".format(
        checkpoint_file
    )
    with pathmgr.open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")

    # extract state_dict from checkpoint
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif 'r3m' in checkpoint:
        state_dict = checkpoint["r3m"]
    elif 'vip' in checkpoint:
        state_dict = checkpoint["vip"]
    else:
        state_dict = checkpoint

    # rename moco-v2 pre-trained keys
    if checkpoint_file.split('/')[-1] in ['moco_v2_800ep_pretrain.pth.tar',
                                          'moco_v2_200ep_pretrain.pth.tar',
                                          'moco_v1_200ep_pretrain.pth.tar']:
        if hasattr(model, 'fc'):
            for k in list(state_dict.keys()):
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                del state_dict[k]
        else:
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

    # rename swav pre-trained keys
    if checkpoint_file.split('/')[-1] in ['swav_800ep_pretrain.pth.tar',
                                          'swav_400ep_pretrain.pth.tar',
                                          'swav_200ep_pretrain.pth.tar',
                                          'swav_100ep_pretrain.pth.tar']:
        for k in list(state_dict.keys()):
            if k.startswith('module.') and not k.startswith('module.projection_head') and not k.startswith('module.prototypes'):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    # rename simsiam or pixpro pre-trained keys
    if checkpoint_file.split('/')[-1] in ['checkpoint_0099.pth.tar',
                                          'pixpro_base_r50_400ep_md5_919c6612.pth']:
        if hasattr(model, 'fc'):
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder.'):
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                del state_dict[k]
        else:
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder.') and not k.startswith('module.encoder.fc'):
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                del state_dict[k]

    # rename R3M and VIP pre-trained keys
    if checkpoint_file.split('/')[-1] in ['original_r3m.pt', 'original_r3m_nol1.pt',
                                          'original_r3m_noaug.pt', 'original_r3m_nolang.pt',
                                          'vip.pt']:
        for k in list(state_dict.keys()):
            if k.startswith('module.convnet.'):
                # remove prefix
                state_dict[k[len("module.convnet."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    # Densecl, VFS don't need to rename keys
    r = unwrap_model(model).load_state_dict(state_dict, strict=False)
    if r.unexpected_keys or r.missing_keys:
        print(f"Loading weights, unexpected keys: {r.unexpected_keys}")
        print(f"Loading weights, missing keys: {r.missing_keys}")
    else:
        print("All keys matched successfully!")


def get_unsupervised_resnet(root_dir, embedding_name, remove_layers=[], encode_stackframes=False):

    pretrain_fname = RESNET_MODELS[embedding_name]
    pretrain_path = os.path.join(root_dir, 'pretrained', pretrain_fname)

    arch = embedding_name.split('-')[-1]
    if arch == 'resnet50':
        model = resnet50(pretrained=False, encode_stackframes=encode_stackframes)
    elif arch == 'resnet50_VIPfc':
        model = resnet50(pretrained=False, encode_stackframes=encode_stackframes)
        setattr(model, 'fc', nn.Linear(2048, 1024))
    else:
        raise NotImplementedError

    # modify arch (remove layer)
    model.modify(remove_layers=remove_layers)

    # infer hidden dim
    model.infer_dims()
    hidden_dim = model.enc_hid_dim

    load_checkpoint(pretrain_path, model)
    print("Loaded encoder from: {}".format(pretrain_path))

    return model, hidden_dim