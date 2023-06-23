import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.vit import get_vit
from agent.resnet import get_unsupervised_resnet
from rewarder import optimal_transport_plan, cosine_distance, euclidean_distance
import copy


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

	def forward(self, obs, std):
		h = self.trunk(obs)

		mu = self.policy(h)
		mu = torch.tanh(mu)
		std = torch.ones_like(mu) * std

		dist = utils.TruncatedNormal(mu, std)
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


class POTILAgent:
	def __init__(self, root_dir, obs_shape, action_shape, device, lr, feature_dim,
				 hidden_dim, critic_target_tau, num_expl_steps,
				 update_every_steps, stddev_schedule, stddev_clip, use_tb, augment,
				 rewards, sinkhorn_rew_scale, update_target_every, use_trunk_target, trunk_target_teacher,
				 auto_rew_scale, auto_rew_scale_factor, suite_name, obs_type, bc_weight_type, bc_weight_schedule,
				 backbone, embedding_name, freeze, fp16, use_encoded_repr,
				 temperature_scaled, temperature, cost2reward, beta):

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
		self.rewards = rewards
		self.sinkhorn_rew_scale = sinkhorn_rew_scale
		self.use_trunk_target = use_trunk_target
		self.update_target_every = update_target_every
		self.trunk_target_teacher = trunk_target_teacher
		self.auto_rew_scale = auto_rew_scale
		self.auto_rew_scale_factor = auto_rew_scale_factor
		self.use_encoder = True if obs_type == 'pixels' else False
		self.bc_weight_type = bc_weight_type
		self.bc_weight_schedule = bc_weight_schedule
		self.fp16 = fp16
		self.encode_repr = not use_encoded_repr

		self.temperature_scaled = temperature_scaled
		self.temperature = temperature
		self.cost2reward = cost2reward
		self.beta = beta

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

		self.repr_dim = repr_dim
		self.freeze_encoder = freeze

		if self.use_trunk_target:
			self.trunk_target = nn.Sequential(
				nn.Linear(repr_dim, feature_dim),
				nn.LayerNorm(feature_dim), nn.Tanh()).to(device)
		else:
			self.trunk_target = None

		self.actor = Actor(repr_dim, action_shape, feature_dim,
						   hidden_dim).to(device)

		self.critic = Critic(repr_dim, action_shape, feature_dim,
							 hidden_dim).to(device)
		self.critic_target = Critic(repr_dim, action_shape, feature_dim,
									hidden_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=6)

		self.train(training=False)
		self.critic_target.train(False)
		if self.use_trunk_target:
			self.trunk_target.train(False)

	def __repr__(self):
		return "potil"

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
		obs = torch.as_tensor(obs, device=self.device)

		if self.fp16:
			with torch.cuda.amp.autocast(enabled=True):
				obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder and self.encode_repr else obs.unsqueeze(0)
			obs = obs.float()
		else:
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

	def update_actor(self, obs, obs_bc, obs_qfilter, action_bc, bc_regularize, step):
		metrics = dict()

		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(obs, stddev)
		action = dist.sample(clip=self.stddev_clip)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		Q1, Q2 = self.critic(obs, action)
		Q = torch.min(Q1, Q2)

		# Compute bc weight
		if not bc_regularize:
			bc_weight = 0.0
		elif self.bc_weight_type == "linear":
			bc_weight = utils.schedule(self.bc_weight_schedule, step)
		elif self.bc_weight_type == "qfilter":
			"""
			Soft Q-filtering inspired from 			
			Nair, Ashvin, et al. "Overcoming exploration in reinforcement 
			learning with demonstrations." 2018 IEEE international 
			conference on robotics and automation (ICRA). IEEE, 2018.
			"""
			with torch.no_grad():
				stddev = 0.1
				dist_qf = self.actor_bc(obs_qfilter, stddev)
				action_qf = dist_qf.mean
				Q1_qf, Q2_qf = self.critic(obs_qfilter.clone(), action_qf)
				Q_qf = torch.min(Q1_qf, Q2_qf)
				bc_weight = (Q_qf > Q).float().mean().detach()

		actor_loss = - Q.mean() * (1 - bc_weight)

		if bc_regularize:
			stddev = 0.1
			dist_bc = self.actor(obs_bc, stddev)
			log_prob_bc = dist_bc.log_prob(action_bc).sum(-1, keepdim=True)
			actor_loss += - log_prob_bc.mean() * bc_weight * 0.03

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()
		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_logprob'] = log_prob.mean().item()
			metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
			metrics['actor_q'] = Q.mean().item()
			if bc_regularize and self.bc_weight_type == "qfilter":
				metrics['actor_qf'] = Q_qf.mean().item()
			metrics['bc_weight'] = bc_weight
			metrics['regularized_rl_loss'] = - Q.mean().item() * (1 - bc_weight)
			metrics['rl_loss'] = - Q.mean().item()
			if bc_regularize:
				metrics['regularized_bc_loss'] = - log_prob_bc.mean().item() * bc_weight * 0.03
				metrics['bc_loss'] = - log_prob_bc.mean().item() * 0.03
		
		return metrics

	def update(self, replay_iter, expert_replay_iter, step, bc_regularize=False):
		metrics = dict()

		if step % self.update_every_steps != 0:
			return metrics

		batch = next(replay_iter)
		obs, action, reward, discount, next_obs = utils.to_torch(
			batch, self.device)

		# augment
		if self.use_encoder and self.augment:
			obs_qfilter = self.aug(obs.clone().float())
			obs = self.aug(obs.float())
			next_obs = self.aug(next_obs.float())
		else:
			obs_qfilter = obs.clone().float()
			obs = obs.float()
			next_obs = next_obs.float()

		if self.use_encoder and self.encode_repr:
			# encode
			with torch.cuda.amp.autocast(enabled=self.fp16):
				obs = self.encoder(obs)
				with torch.no_grad():
					next_obs = self.encoder(next_obs)
			obs = obs.float()
			next_obs = next_obs.float()

		if bc_regularize:
			batch = next(expert_replay_iter)
			obs_bc, action_bc = utils.to_torch(batch, self.device)
			# augment
			if self.use_encoder and self.augment:
				obs_bc = self.aug(obs_bc.float())
			else:
				obs_bc = obs_bc.float()
			# encode
			if bc_regularize and self.bc_weight_type == "qfilter":
				if self.fp16:
					with torch.cuda.amp.autocast():
						obs_qfilter = self.encoder_bc(obs_qfilter) if self.use_encoder and self.encode_repr else obs_qfilter
					obs_qfilter = obs_qfilter.float()
				else:
					obs_qfilter = self.encoder_bc(obs_qfilter) if self.use_encoder and self.encode_repr else obs_qfilter
				obs_qfilter = obs_qfilter.detach()
			else:
				obs_qfilter = None

			if self.fp16:
				with torch.cuda.amp.autocast():
					obs_bc = self.encoder(obs_bc) if self.use_encoder and self.encode_repr else obs_bc
				obs_bc = obs_bc.float()
			else:
				obs_bc = self.encoder(obs_bc) if self.use_encoder and self.encode_repr else obs_bc
			# Detach grads
			obs_bc = obs_bc.detach()
		else:
			obs_qfilter = None
			obs_bc = None 
			action_bc = None

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(
			self.update_critic(obs, action, reward, discount, next_obs, step))

		# update actor
		metrics.update(self.update_actor(obs.detach(), obs_bc, obs_qfilter, action_bc, bc_regularize, step))

		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics

	def bc_update(self, expert_replay_iter, step):
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

	def ot_rewarder(self, observations, demos, step):

		if step % self.update_target_every == 0:
			if self.use_trunk_target:
				if self.trunk_target_teacher == 'actor':
					self.trunk_target.load_state_dict(self.actor.trunk.state_dict())
				elif self.trunk_target_teacher == 'critic':
					self.trunk_target.load_state_dict(self.critic.trunk.state_dict())
				else:
					raise NotImplementedError
				self.target_updated = True

		scores_list = list()
		ot_rewards_list = list()
		for demo in demos:

			obs = torch.tensor(observations).to(self.device).float()
			if self.use_encoder and self.encode_repr:
				with torch.cuda.amp.autocast(enabled=self.fp16):
					obs = self.encoder(obs)
				obs = obs.float()
			if self.use_trunk_target:
				obs = self.trunk_target(obs)

			exp = torch.tensor(demo).to(self.device).float()
			if self.use_encoder and self.encode_repr:
				with torch.cuda.amp.autocast(enabled=self.fp16):
					exp = self.encoder(exp)
				exp = exp.float()
			if self.use_trunk_target:
				exp = self.trunk_target(exp)

			obs = obs.detach()
			exp = exp.detach()

			if self.rewards == 'sinkhorn_cosine':
				cost_matrix = cosine_distance(obs, exp)  # Get cost matrix for samples using critic network.

				if self.temperature_scaled:
					cost_matrix = torch.exp(cost_matrix / self.temperature)

				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn', niter=100).float()  # Getting optimal coupling

				if self.cost2reward == 'linear':
					ot_rewards = - self.sinkhorn_rew_scale * torch.diag(
						torch.mm(transport_plan, cost_matrix.T)).detach().cpu().numpy()
				elif self.cost2reward == 'exp':
					cost_sum = torch.diag(torch.mm(transport_plan, cost_matrix.T))
					ot_rewards = self.sinkhorn_rew_scale * torch.exp(-self.beta * cost_sum)
					ot_rewards = ot_rewards.detach().cpu().numpy()
				else:
					raise NotImplementedError
				
			elif self.rewards == 'sinkhorn_euclidean':
				cost_matrix = euclidean_distance(
					obs, exp)  # Get cost matrix for samples using critic network.
				transport_plan = optimal_transport_plan(
					obs, exp, cost_matrix, method='sinkhorn', niter=100, epsilon=0.1).float()  # Getting optimal coupling
				ot_rewards = - self.sinkhorn_rew_scale * torch.diag(
					torch.mm(transport_plan,
							 cost_matrix.T)).detach().cpu().numpy()
				
			elif self.rewards == 'cosine':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(1. - F.cosine_similarity(obs, exp))
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()
				
			elif self.rewards == 'euclidean':
				exp = torch.cat((exp, exp[-1].unsqueeze(0)))
				ot_rewards = -(obs - exp).norm(dim=1)
				ot_rewards *= self.sinkhorn_rew_scale
				ot_rewards = ot_rewards.detach().cpu().numpy()
				
			else:
				raise NotImplementedError()

			scores_list.append(np.sum(ot_rewards))
			ot_rewards_list.append(ot_rewards)

		closest_demo_index = np.argmax(scores_list)
		return ot_rewards_list[closest_demo_index]

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
		if self.use_trunk_target:
			if self.trunk_target_teacher == 'actor':
				self.trunk_target.load_state_dict(self.actor.trunk.state_dict())
			elif self.trunk_target_teacher == 'critic':
				self.trunk_target.load_state_dict(self.critic.trunk.state_dict())
			else:
				raise NotImplementedError

		if self.bc_weight_type == "qfilter":
			# Store a copy of the BC policy with frozen weights
			if self.use_encoder:
				self.encoder_bc = copy.deepcopy(self.encoder)
				for param in self.encoder_bc.parameters():
					param.requires_grad = False
				self.encoder_bc.train(False)

			self.actor_bc = copy.deepcopy(self.actor)
			for param in self.actor_bc.parameters():
				param.required_grad = False
			self.actor_bc.train(False)

		# Update optimizers
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

	def load_stage1(self):
		if self.use_trunk_target:
			if self.trunk_target_teacher == 'actor':
				self.trunk_target.load_state_dict(self.actor.trunk.state_dict())
			elif self.trunk_target_teacher == 'critic':
				self.trunk_target.load_state_dict(self.critic.trunk.state_dict())
			else:
				raise NotImplementedError

		if self.bc_weight_type == "qfilter":
			# Store a copy of the BC policy with frozen weights
			if self.use_encoder:
				self.encoder_bc = copy.deepcopy(self.encoder)
				for param in self.encoder_bc.parameters():
					param.requires_grad = False
				self.encoder_bc.train(False)

			self.actor_bc = copy.deepcopy(self.actor)
			for param in self.actor_bc.parameters():
				param.required_grad = False
			self.actor_bc.train(False)

		# Update optimizers
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
