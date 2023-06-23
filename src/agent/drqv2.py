import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.vit import get_vit
from agent.resnet import get_unsupervised_resnet


class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
		super().__init__()

		self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())

		self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, action_shape[0]))

		self.apply(utils.weight_init)

	def forward(self, obs, std, return_pretanh=False):
		h = self.trunk(obs)

		mu = self.policy(h)
		pretanh = mu
		mu = torch.tanh(mu)
		std = torch.ones_like(mu) * std

		dist = utils.TruncatedNormal(mu, std)

		if return_pretanh:
			return (dist, pretanh)
		else:
			return dist


class Critic(nn.Module):
	def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
		super().__init__()

		self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())

		self.Q1 = nn.Sequential(
			nn.Linear(feature_dim + action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.Q2 = nn.Sequential(
			nn.Linear(feature_dim + action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.apply(utils.weight_init)

	def forward(self, obs, action):
		h = self.trunk(obs)
		h_action = torch.cat([h, action], dim=-1)
		q1 = self.Q1(h_action)
		q2 = self.Q2(h_action)

		return q1, q2


class DrQv2Agent:
	def __init__(self, root_dir, obs_shape, action_shape, device, lr, feature_dim,
				 hidden_dim, critic_target_tau, num_expl_steps,
				 update_every_steps, stddev_schedule, stddev_clip, use_tb,
				 augment, obs_type, use_encoded_repr,
				 backbone, embedding_name, freeze, fp16):

		self.root_dir = root_dir
		self.device = device
		self.lr = lr
		self.critic_target_tau = critic_target_tau
		self.update_every_steps = update_every_steps
		self.use_tb = use_tb
		self.num_expl_steps = num_expl_steps
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.augment = augment
		self.use_encoder = True if obs_type == 'pixels' else False
		self.fp16 = fp16
		self.encode_repr = not use_encoded_repr

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

		self.repr_dim = repr_dim
		self.freeze_encoder = freeze

		self.actor = Actor(repr_dim, action_shape, feature_dim,
						   hidden_dim).to(device)

		self.critic = Critic(repr_dim, action_shape, feature_dim,
							 hidden_dim).to(device)
		self.critic_target = Critic(repr_dim, action_shape,
									feature_dim, hidden_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		# optimizers
		if self.use_encoder and not self.freeze_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=6)

		self.train(training=False)
		self.critic_target.train(False)

	def __repr__(self):
		return "drqv2"
	
	def train(self, training=True):
		self.training = training
		if self.use_encoder and not self.freeze_encoder:
			self.encoder.train(training)
		self.actor.train(training)
		self.critic.train(training)

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
		obs = torch.as_tensor(obs, device=self.device).float()

		obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder and self.encode_repr else obs.unsqueeze(0)

		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(obs, stddev)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample(clip=None)
			if step < self.num_expl_steps:
				action.uniform_(-1.0, 1.0)
		return action.cpu().numpy()[0]

	def update_critic(self, obs, action, reward, discount, next_obs, step):
		metrics = dict()

		with torch.no_grad():
			stddev = utils.schedule(self.stddev_schedule, step)
			dist = self.actor(next_obs, stddev)
			next_action = dist.sample(clip=self.stddev_clip)
			target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
			target_V = torch.min(target_Q1, target_Q2)
			target_Q = reward + (discount * target_V)

		Q1, Q2 = self.critic(obs, action)

		critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

		# optimize encoder and critic
		if self.use_encoder and not self.freeze_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_opt.step()
		if self.use_encoder and not self.freeze_encoder:
			self.encoder_opt.step()

		if self.use_tb:
			metrics['critic_target_q'] = target_Q.mean().item()
			metrics['critic_q1'] = Q1.mean().item()
			metrics['critic_q2'] = Q2.mean().item()
			metrics['critic_loss'] = critic_loss.item()
			
		return metrics

	def update_actor(self, obs, step):
		metrics = dict()

		stddev = utils.schedule(self.stddev_schedule, step)
		dist, pretanh = self.actor(obs, stddev, return_pretanh=True)
		action = dist.sample(clip=self.stddev_clip)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		Q1, Q2 = self.critic(obs, action)
		Q = torch.min(Q1, Q2)

		actor_loss = -Q.mean()

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()

		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_logprob'] = log_prob.mean().item()
			metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

		return metrics

	def update(self, replay_iter, step):
		metrics = dict()

		if step % self.update_every_steps != 0:
			return metrics

		batch = next(replay_iter)
		obs, action, reward, discount, next_obs = utils.to_torch(
			batch, self.device)

		# augment
		if self.use_encoder and self.augment:
			obs = self.aug(obs.float())
			next_obs = self.aug(next_obs.float())
		else:
			obs = obs.float()
			next_obs = next_obs.float()

		# encode
		if self.use_encoder and self.encode_repr:
			obs = self.encoder(obs)
			with torch.no_grad():
				next_obs = self.encoder(next_obs)

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(obs, action, reward, discount, next_obs, step))

		# update actor
		metrics.update(self.update_actor(obs.detach(), step))
			
		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics

	def save_snapshot(self):
		keys_to_save = ['actor', 'critic']
		if self.use_encoder:
			keys_to_save += ['encoder']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload):
		for k, v in payload.items():
			self.__dict__[k] = v
		self.critic_target.load_state_dict(self.critic.state_dict())
		# Update optimizers
		if self.use_encoder and not self.freeze_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)


