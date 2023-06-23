import numpy as np
import torch
from torch import nn, optim, distributions

import utils
from agent.vit import get_vit
from agent.resnet import get_unsupervised_resnet


class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape):
		super().__init__()

		policy_list = [nn.LayerNorm(repr_dim)]
		policy_list.append(nn.Linear(repr_dim, 512))
		policy_list.append(nn.ReLU(inplace=True))
		policy_list.append(nn.Linear(512, 256))
		policy_list.append(nn.ReLU(inplace=True))
		policy_list.append(nn.Linear(256, 128))
		policy_list.append(nn.ReLU(inplace=True))
		policy_list.append(nn.Linear(128, action_shape[0]))

		self.policy = nn.Sequential(*policy_list)
		self.apply(utils.weight_init)

	def forward(self, obs, std):
		mu = self.policy(obs)
		mu = torch.tanh(mu)
		std = torch.ones_like(mu) * std

		dist = utils.TruncatedNormal(mu, std)
		return dist


class BCAgent:
	def __init__(self, root_dir, obs_shape, action_shape, device, lr, feature_dim,
				 hidden_dim, stddev_schedule, stddev_clip, use_tb, augment, suite_name, obs_type,
				 backbone, embedding_name, freeze, fp16, use_encoded_repr):

		self.root_dir = root_dir
		self.device = device
		self.lr = lr
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.use_tb = use_tb
		self.augment = augment
		self.fp16 = fp16
		self.encode_repr = not use_encoded_repr
		self.use_encoder = True if obs_type == 'pixels' else False

		# models
		if self.use_encoder:
			if backbone == 'vit':
				self.encoder, repr_dim = get_vit(root_dir, embedding_name, obs_shape[-1])
				self.encoder.to(device)
			elif backbone == 'resnet':
				self.encoder, repr_dim = get_unsupervised_resnet(root_dir, embedding_name)
				self.encoder.to(device)
			else:
				raise NotImplementedError

			if freeze:
				self.encoder.freeze()
				self.encoder.eval()
		else:
			repr_dim = obs_shape[0]

		self.freeze_encoder = freeze

		self.actor = Actor(repr_dim, action_shape).to(device)

		# optimizers
		if self.use_encoder and not self.freeze_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=6)

		self.train(False)

	def __repr__(self):
		return "bc"
	
	def train(self, training=True):
		self.training = training
		if self.use_encoder and not self.freeze_encoder:
			self.encoder.train(training)
		self.actor.train(training)

	def get_repr(self, obs):
		obs = torch.as_tensor(obs, device=self.device)
		obs_shape = obs.shape
		if len(obs_shape) == 3:
			obs = obs.unsqueeze(0)

		with torch.cuda.amp.autocast(enabled=self.fp16):
			repr = self.encoder(obs)
		repr = repr.float().cpu().numpy()

		if len(obs_shape) == 3:
			return repr[0]
		else:
			return repr

	def act(self, obs, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device)

		obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder and self.encode_repr else obs.unsqueeze(0)

		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(obs, stddev)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample(clip=None)
		return action.cpu().numpy()[0]

	def update(self, expert_replay_iter, step):
		metrics = dict()

		batch = next(expert_replay_iter)
		obs, action = utils.to_torch(batch, self.device)
		action = action.float()

		if self.use_encoder and self.augment:
			obs = self.aug(obs.float())
		else:
			obs = obs.float()

		if self.use_encoder and self.encode_repr:
			with torch.cuda.amp.autocast(enabled=self.fp16):
				obs = self.encoder(obs)
			obs = obs.float()

		stddev = utils.schedule(self.stddev_schedule, step)
		dist = self.actor(obs, stddev)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		actor_loss = - log_prob.mean()

		if self.use_encoder and not self.freeze_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		if self.use_encoder and not self.freeze_encoder:
			self.encoder_opt.step()
		self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_logprob'] = log_prob.mean().item()
			metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

		return metrics

	def save_snapshot(self):
		keys_to_save = ['actor']
		if self.use_encoder:
			keys_to_save += ['encoder']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload):
		for k, v in payload.items():
			self.__dict__[k] = v

		# Update optimizers
		if self.use_encoder and not self.freeze_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
