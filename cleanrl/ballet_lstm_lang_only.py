from cleanrl.ballet_env import ballet_environment
from cleanrl.ballet_env.ballet_wrappers import BalletEnv, SyncVectorBalletEnv

import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default='2_delay16',
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def lstm_init(lstm):
    for name, param in lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
    return lstm

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Encoder block
        self.img_encoder = nn.Sequential(
            layer_init(nn.Conv2d(3, 16, 9, stride=9)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 7 * 7, 256)),
            nn.ReLU(),
        )
        self.lang_encoder_lstm = nn.LSTM(14, 256)
        self.lang_encoder_lstm = lstm_init(self.lang_encoder_lstm)
        self.lang_embedding = nn.Sequential(
            layer_init(nn.Linear(256, 32)),
            nn.ReLU(),
        )
        # Memory block
        self.memory_lstm = nn.LSTM(256+32, 512, 4)
        self.memory_lstm = lstm_init(self.memory_lstm)

        # Decoder block
        self.actor = layer_init(nn.Linear(512, 8), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        
    def get_states(self, x, lstm_state_dict, done):
        # Encoder logic
        img_hidden = self.img_encoder(x[0])
        batch_size = lstm_state_dict["encoder"][0].shape[1]
        lang_input = x[1].reshape((-1, batch_size, self.lang_encoder_lstm.input_size))
        lang_hidden = []
        for h, d in zip(lang_input, done):
            h, lstm_state_dict["encoder"] = self.lang_encoder_lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state_dict["encoder"][0],
                    (1.0 - d).view(1, -1, 1) * lstm_state_dict["encoder"][1],
                ),
            )
            lang_hidden += [h]
        lang_hidden = torch.flatten(torch.cat(lang_hidden), 0, 1)
        lang_hidden = self.lang_embedding(lang_hidden)
        hidden = torch.cat([img_hidden, lang_hidden], 1)

        # Memory logic
        batch_size = lstm_state_dict["memory"][0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.memory_lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state_dict["memory"] = self.memory_lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state_dict["memory"][0],
                    (1.0 - d).view(1, -1, 1) * lstm_state_dict["memory"][1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state_dict

    def get_value(self, x, lstm_state_dict, done):
        hidden, _ = self.get_states(x, lstm_state_dict, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state_dict, done, action=None):
        # encoder and memory
        hidden, lstm_state_dict = self.get_states(x, lstm_state_dict, done)

        # actor output
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        # critic output
        value = self.critic(hidden)

        return action, probs.log_prob(action), probs.entropy(), value, lstm_state_dict



if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = SyncVectorBalletEnv(
        [BalletEnv(ballet_environment.simple_builder(level_name=args.env_id)) for i in range(args.num_envs)]
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs_img = torch.zeros((args.num_steps, args.num_envs) + (3, 99, 99)).to(device) # (3, 99, 99) for image observation space of ballet
    obs_lang = torch.zeros((args.num_steps, args.num_envs, 14)).to(device) # 14 for language observation space of ballet
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    (next_obs_img, next_obs_lang) = envs.reset()
    next_obs_img, next_obs_lang = torch.Tensor(next_obs_img).to(device), torch.Tensor(next_obs_lang).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state_dict = {}
    next_lstm_state_dict["encoder"] = (
        torch.zeros(agent.lang_encoder_lstm.num_layers, args.num_envs, agent.lang_encoder_lstm.hidden_size).to(device),
        torch.zeros(agent.lang_encoder_lstm.num_layers, args.num_envs, agent.lang_encoder_lstm.hidden_size).to(device),
    )
    next_lstm_state_dict["memory"] = (
        torch.zeros(agent.memory_lstm.num_layers, args.num_envs, agent.memory_lstm.hidden_size).to(device),
        torch.zeros(agent.memory_lstm.num_layers, args.num_envs, agent.memory_lstm.hidden_size).to(device),
    )
    # next_lstm_state_dict["decoder"] = (
    #     torch.zeros(agent.lang_decoder_lstm.num_layers, args.num_envs, agent.lang_decoder_lstm.hidden_size).to(device),
    #     torch.zeros(agent.lang_decoder_lstm.num_layers, args.num_envs, agent.lang_decoder_lstm.hidden_size).to(device),
    # )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        initial_lstm_state_dict = {}
        for key in next_lstm_state_dict.keys():
            initial_lstm_state_dict[key] = (next_lstm_state_dict[key][0].clone(), next_lstm_state_dict[key][1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs_img[step] = next_obs_img
            obs_lang[step] = next_obs_lang
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state_dict = agent.get_action_and_value((next_obs_img, next_obs_lang), next_lstm_state_dict, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            (next_obs_img, next_obs_lang), reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs_img, next_obs_lang = torch.Tensor(next_obs_img).to(device), torch.Tensor(next_obs_lang).to(device)
            next_done = torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    avg_returns.append(item['episode']['r'])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                (next_obs_img, next_obs_lang),
                next_lstm_state_dict,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs_img = obs_img.reshape((-1,) + (3, 99, 99))
        b_obs_lang = obs_lang.reshape((-1, 14))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                # cut lstm
                lstm_dict_for_train = {}
                for key in next_lstm_state_dict.keys():
                    lstm_dict_for_train[key] = (initial_lstm_state_dict[key][0][:, mbenvinds], initial_lstm_state_dict[key][1][:, mbenvinds])

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    (b_obs_img[mb_inds], b_obs_lang[mb_inds]),
                    lstm_dict_for_train,
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()


                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()