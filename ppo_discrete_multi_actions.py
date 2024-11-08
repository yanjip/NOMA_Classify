# time: 2023/10/30 18:09
# author: YanJP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from replaybuffer import ReplayBuffer
from torch.distributions import Categorical
import para
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

    return layer


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.actor_rnn_hidden = None
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        # self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.actor_rnn = nn.GRU(args.hidden_width, args.hidden_width, batch_first=True)
        self.output_layers = nn.ModuleList([
            nn.Linear(args.hidden_width, num_actions) for num_actions in para.output_dims
        ])
        # self.fc3 = nn.Linear(args.hidden_width, args.action_dim)

        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.actor_rnn)
            # orthogonal_init(self.fc3, gain=0.01)


    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        output, self.actor_rnn_hidden = self.actor_rnn(s, self.actor_rnn_hidden)
        s = self.activate_func(output)
        # a_prob = torch.softmax(self.fc3(s), dim=1)  #这是一个所有动作对应的概率
        # return a_prob
        categorical_objects = [Categorical(logits=output_layer(s)) for output_layer in self.output_layers]
        return categorical_objects

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.critic_rnn = nn.GRU(args.hidden_width, args.hidden_width, batch_first=True)
        self.critic_rnn_hidden = None
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.critic_rnn)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        # s = self.activate_func(self.critic_rnn)
        output, self.critic_rnn_hidden = self.critic_rnn(s, self.critic_rnn_hidden)
        s=self.activate_func(output)
        v_s = self.fc3(s)
        return v_s


class PPO_discrete:
    def __init__(self, args):
        self.device=device
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.actor = Actor(args).to(self.device)
        self.critic = Critic(args).to(self.device)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
    def reset_rnn_hidden(self):
        self.actor.actor_rnn_hidden = None
        self.critic.critic_rnn_hidden = None
    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
        # a_prob = self.actor(s).detach().cpu().numpy().flatten()
        # a = np.argmax(a_prob)
        # return a
        actions=[]
        action_logprobs = []
        categorical_objects = self.actor(s)
        for i, categorical in enumerate(categorical_objects):
            a_prob = torch.softmax(categorical.logits, dim=1).detach().cpu().numpy().flatten()
            a=np.argmax(a_prob)
            actions.append(a)
        return actions

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        with torch.no_grad():
        #     dist = Categorical(probs=self.actor(s))
        #     a = dist.sample()
        #     a_logprob = dist.log_prob(a)
        # return a.cpu().numpy()[0], a_logprob.cpu().numpy()[0]
            actions=[]
            action_logprobs = []
            categorical_objects=self.actor(s)
            for i, categorical in enumerate(categorical_objects):
                a = categorical.sample()
                a_logprob = categorical.log_prob(a)
                actions.append(a.cpu().numpy()[0])
                action_logprobs.append(a_logprob.cpu().numpy()[0])
                # print(f"Sampled action for dimension {i + 1}: {sampled_action.item()}, Log probability: {log_prob.item()}")
        return actions, action_logprobs

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        s=s.to(self.device)
        a=a.to(self.device)
        a_logprob=a_logprob.to(self.device)
        r=r.to(self.device)
        s_=s_.to(self.device)
        dw=dw.to(self.device)
        done=done.to(self.device)

        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(device)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now=0
                self.reset_rnn_hidden()
                categorical_objects = self.actor(s[index])
                actor_loss=torch.zeros(self.mini_batch_size,1).to(device)
                for i, categorical in enumerate(categorical_objects):
                    a_prob = torch.softmax(categorical.logits, dim=1)
                    dist_now=Categorical(probs=a_prob)
                # dist_now = Categorical(probs=self.actor(s[index]))
                    dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                    a_logprob_now = dist_now.log_prob(a[index][:,i].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                    # a/b=exp(log(a)-log(b))
                    ratios = torch.exp(a_logprob_now - a_logprob[index][:,i].squeeze().view(-1, 1))   # shape(mini_batch_size X 1)

                    surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                    actor_loss += -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                # print(_,'loss:',actor_loss.item())
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def savemodel(self,time,path=None):
        path = 'runs/model/ppo_' + time + '.pth'
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

