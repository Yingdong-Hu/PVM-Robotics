# For Pre-Trained Vision Models in Motor Control, Not All Policy Learning Methods are Created Equal
This is a repository containing the code for the paper:

[For Pre-Trained Vision Models in Motor Control, Not All Policy Learning Methods are Created Equal](https://arxiv.org/abs/2304.04591). ICML 2023
Yingdong Hu, Renhao Wang, Li Erran Li, and Yang Gao

![](https://p.ipic.vip/ku33dn.png)

## Installation

### Dependency Setup

- **Install the following libraries**
```
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```
- **Set up Environment**
```
conda env create -f conda_env.yml
conda activate pvm
```

- Install [PyTorch](https://pytorch.org/), torchvision and timm following official instructions. For example:

```
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install timm==0.4.5
```

- **Install [MuJoCo](http://www.mujoco.org/) version 2.1 and mujoco-py**

1. Please follow the [instructions](https://github.com/openai/mujoco-py#install-mujoco) in the mujoco-py package. 
2. You should make sure that the GPU version of mujoco-py gets built, so that image rendering is fast. An easy way to ensure this is to clone the mujoco-py repository, change [this line](https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/builder.py#L74) to `Builder = LinuxGPUExtensionBuilder`, and install from source by running `pip install -e .` in the `mujoco-py` root directory. You can also download our [changed mujoco-py package](https://drive.google.com/file/d/1WDUEs1ladlO7iwlRzcVUEjU49QZr32ES/view?usp=sharing) and install from source.

- **Install Meta-World**

Download the package from [here](https://drive.google.com/file/d/1Gv7qQNWzXjBs-Zek9AdG991qyzB16dJm/view?usp=sharing).
```
pip install -e /path/to/dir/metaworld
```
- **Install Robosuite**

We use the `offline_study` branch of Robosuite, dowload it from [here](https://drive.google.com/file/d/1EGjUrAPuLJ4vSbMSmOt322fzN1teH69d/view?usp=sharing).
```
pip install -e /path/to/dir/robosuite-offline_study
```

- **Install Franka-Kitchen**

Please follow the [instructions](https://github.com/facebookresearch/r3m/tree/76a7a9eeeca6f034c12a9ffad4425ea36f4a139a/evaluation) in the R3M repository.

### Download Pre-Trained Vision Models

| Model | Architecture | Highlights | Link |  
| :---: | :---: | :---: |:---: |
| MoCo v2 | ResNet-50 | Contrastive learning, momentum encoder | [download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/moco_v2_800ep_pretrain.pth.tar) |
| SwAV | ResNet-50 | Contrast online cluster assignments | [download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/swav_800ep_pretrain.pth.tar) |
| SimSiam | ResNet-50 | Without negative pairs | [download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/checkpoint_0099.pth.tar) |
| DenseCL | ResNet-50 | Dense contrastive learning, learn local features | [download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/densecl_r50_imagenet_200ep.pth) |
| PixPro | ResNet-50 | Pixel-level pretext task, learn local features |[download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/pixpro_base_r50_400ep_md5_919c6612.pth) |
| VICRegL | ResNet-50 | Learn global and local features | [download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/resnet50_alpha0.9.pth) |
| VFS | ResNet-50 | Encode temporal dynamics | [download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/r50_nc_sgd_cos_100e_r5_1xNx2_k400-d7ce3ad0.pth) |
| R3M | ResNet-50 | Learn visual representations for robotics | [download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/original_r3m.pt) |
| VIP | ResNet-50 | Learn representations and reward for robotics | [download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/vip.pt) |
| MoCo v3 | ViT-B/16 | Contrastive learning for ViT | [download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/vit-b-300ep.pth.tar) |
| DINO | ViT-B/16 | Self-distillation with no labels | [download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/dino_vitbase16_pretrain.pth) |
| MAE | ViT-B/16 | Masked image modeling (MIM) | [download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/mae_pretrain_vit_base.pth) |
| iBOT | ViT-B/16 | Combine self-distillation with MIM | [download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/ibot-vit-b16_checkpoint_teacher.pth) |
| CLIP | ViT-B/16 | Language-supervised pre-training | [download](https://github.com/Yingdong-Hu/PVM-Robotics/releases/download/v1.0.0/CLIP-ViT-B-16.pt) |

After downloading a pre-trained vision model, place it under `PVM-Robotics/pretrained/` folder. Please don't modify the file names of these checkpoints.

### Download Expert Demonstrations

- Download the expert demonstrations for all tasks from [here](https://drive.google.com/file/d/1FLHtXdcnC86n38BEDDN_LVtmF16lgiRh/view?usp=sharing).
- Unzip `expert_demos.zip` and place the `expert_demos` directory into `PVM-Robotics/expert_demos`.
- set the `path/to/dir` portion of the `root_dir` path variable in `cfgs/config.yaml` to the path of the PVM-Robotics repository.

## Train Agents

### Reinforcement learning

#### Meta-World
```
python train_rl.py \
agent=drqv2 \
suite=metaworld \
suite/metaworld_task=hammer \
agent.backbone=resnet \
agent.embedding_name=mocov2-resnet50 \
replay_buffer_size=500000 suite.num_seed_frames=4000 batch_size=512 \
use_wandb=true seed=1 exp_prefix=RL
```
- `suite/metaworld_task` can be set to `hammer`, `drawer_close`, `door_open`, `bin_picking`, `button_press_topdown`, `window_close`, `lever_pull`, and `coffee_pull`.
- When `agent.backbone` is set to `resnet`, `agent.embedding_name` can be set to `mocov2-resnet50`, `simsiam-resnet50`, `swav-resnet50`, `densecl-resnet50`, `pixpro-resnet50`, `vicregl-resnet50`, `vfs-resnet50`, `r3m-resnet50`, and `vip-resnet50_VIPfc`.
- When `agent.backbone` is set to `vit`, `agent.embedding_name` can be set to `mocov3-vit-b16`, `dino-vit-b16`, `ibot-vit-b16`, `clip-vit-b16`, and `mae-vit-b16`.

#### Robosuite
```
python train_rl.py \
agent=drqv2 \
suite=robosuite \
suite/robosuite_task=panda_door \
agent.backbone=resnet \
agent.embedding_name=mocov2-resnet50 \
replay_buffer_size=500000 suite.num_seed_frames=4000 batch_size=512 \
use_wandb=true seed=1 exp_prefix=RL
```
- `suite/robosuite_task` can be set to `panda_door`, `panda_lift`, `panda_twoarm_peginhole`, `panda_pickplace_can`, `panda_nut_assembly_square`, `jaco_door`, `jaco_lift`, and `jaco_twoarm_peginhole`.

#### Franka-Kitchen
```
python train_rl.py \
agent=drqv2 \
suite=kitchen \
suite/kitchen_task=turn_knob \
agent.backbone=resnet \
agent.embedding_name=mocov2-resnet50 \
num_train_frames_drq=1100000 replay_buffer_size=500000 suite.num_seed_frames=4000 batch_size=512 \
use_wandb=true seed=1 exp_prefix=RL
```
- `suite/kitchen_task` can be set to `turn_knob`, `turn_light_on`, `slide_door`, `open_door`, and `open_micro`.
- We train RL agents for 1.1M environment steps on Franka-Kitchen.

### Imitation learning through behavior cloning

#### Meta-World
```
python train_bc.py \
agent=bc \
suite=metaworld \
suite/metaworld_task=hammer \
agent.backbone=resnet \
agent.embedding_name=mocov2-resnet50 \
num_demos=25 \
use_wandb=true seed=1 exp_prefix=BC
```
- For Meta-World, the maximum value of  `num_demos` is 25.

#### Robosuite
```
python train_bc.py \
agent=bc \
suite=robosuite \
suite/robosuite_task=panda_door \
agent.backbone=resnet \
agent.embedding_name=mocov2-resnet50 \
num_demos=50 \
use_wandb=true seed=1 exp_prefix=BC
```
- For Robosuite, the maximum value of  `num_demos` is 50.

#### Franka-Kitchen
```
python train_bc.py \
agent=bc \
suite=kitchen \
suite/kitchen_task=turn_knob \
agent.backbone=resnet \
agent.embedding_name=mocov2-resnet50 \
num_demos=25 \
use_wandb=true seed=1 exp_prefix=BC
```
- For Franka-Kitchen, the maximum value of  `num_demos` is 25.

### Imitation learning with a visual reward function

#### Meta-World
```
python train_vrf.py \
agent=potil \
suite=metaworld \
suite/metaworld_task=hammer \
agent.backbone=resnet \
agent.embedding_name=mocov2-resnet50 \
bc_regularize=true num_demos=1 \
use_wandb=true seed=1 exp_prefix=VRF
```

#### Robosuite
```
python train_vrf.py \
agent=potil \
suite=robosuite \
suite/robosuite_task=panda_door \
agent.backbone=resnet \
agent.embedding_name=mocov2-resnet50 \
bc_regularize=true num_demos=1 \
use_wandb=true seed=1 exp_prefix=VRF
```

#### Franka-Kitchen
```
python train_vrf.py \
agent=potil \
suite=kitchen \
suite/kitchen_task=turn_knob \
agent.backbone=resnet \
agent.embedding_name=mocov2-resnet50 \
bc_regularize=true num_demos=1 \
use_wandb=true seed=1 exp_prefix=VRF
```

## Acknowledgement

We have modified and integrated the code from [ROT](https://github.com/siddhanthaldar/ROT) and [DrQ-v2](https://github.com/facebookresearch/drqv2) into this project.

## Citation

If you find this repository useful, please consider giving a star :star: and citation:
```latex
@article{hu2023pre,
  title={For Pre-Trained Vision Models in Motor Control, Not All Policy Learning Methods are Created Equal},
  author={Hu, Yingdong and Wang, Renhao and Li, Li Erran and Gao, Yang},
  journal={arXiv preprint arXiv:2304.04591},
  year={2023}
}
```