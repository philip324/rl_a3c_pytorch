from __future__ import division
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import itertools, copy

batch_size = 64

def train(rank, args, shared_model, optimizer, env_conf):

    torch.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.model = A3Clstm(player.env.observation_space.shape[0], player.env.action_space)
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    player.model.train()

    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    LOG_EVERY_N_STEPS = 10000
    times, mean_ep_rewards, best_ep_rewards = [], [], []

    for tt in itertools.count():
        player.model.load_state_dict(shared_model.state_dict())
        for step in range(args.num_steps):
            t = tt*args.num_steps+step
            player.action_train(t)
            if (t > 50000 and
                    t % 4 == 0 and
                    player.replay_buffer.can_sample(batch_size)):

                obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask_batch =  player.replay_buffer.sample(batch_size)
                rew_food_t_batch = rew_t_batch[:, 0]
                rew_fruit_t_batch = rew_t_batch[:, 1]
                rew_avoid_t_batch = rew_t_batch[:, 2]
                rew_eat_t_batch = rew_t_batch[:, 3]

                if not model_initialized:
                    initialize_interdependent_variables(session, tf.global_variables(), {
                           obs_t_ph: obs_t_batch,
                           obs_tp1_ph: obs_tp1_batch})
                    session.run(update_target_fn)
                    model_initialized = True

                session.run(train_food_fn, feed_dict={
                                obs_t_ph:obs_t_batch,
                                act_t_ph:act_t_batch,
                                rew_food_t_ph:rew_food_t_batch,
                                obs_tp1_ph:obs_tp1_batch,
                                done_mask_ph:done_mask_batch,
                                learning_rate:optimizer_spec.lr_schedule.value(t)})
                session.run(train_avoid_fn, feed_dict={
                                obs_t_ph:obs_t_batch,
                                act_t_ph:act_t_batch,
                                rew_avoid_t_ph:rew_avoid_t_batch,
                                obs_tp1_ph:obs_tp1_batch,
                                done_mask_ph:done_mask_batch,
                                learning_rate:optimizer_spec.lr_schedule.value(t)})
                session.run(train_fruit_fn, feed_dict={
                                obs_t_ph:obs_t_batch,
                                act_t_ph:act_t_batch,
                                rew_fruit_t_ph:rew_fruit_t_batch,
                                obs_tp1_ph:obs_tp1_batch,
                                done_mask_ph:done_mask_batch,
                                learning_rate:optimizer_spec.lr_schedule.value(t)})
                session.run(train_eat_fn, feed_dict={
                                obs_t_ph:obs_t_batch,
                                act_t_ph:act_t_batch,
                                rew_eat_t_ph:rew_eat_t_batch,
                                obs_tp1_ph:obs_tp1_batch,
                                done_mask_ph:done_mask_batch,
                                learning_rate:optimizer_spec.lr_schedule.value(t)})


                if num_param_updates % target_update_freq == 0:
                    session.run(update_target_fn)
                    train_food_loss = session.run(food_error, feed_dict={obs_t_ph:obs_t_batch,
                                                                         act_t_ph:act_t_batch,
                                                                         rew_food_t_ph:rew_food_t_batch,
                                                                         obs_tp1_ph:obs_tp1_batch,
                                                                         done_mask_ph:done_mask_batch})
                    train_avoid_loss = session.run(avoid_error, feed_dict={obs_t_ph:obs_t_batch,
                                                                         act_t_ph:act_t_batch,
                                                                         rew_avoid_t_ph:rew_avoid_t_batch,
                                                                         obs_tp1_ph:obs_tp1_batch,
                                                                         done_mask_ph:done_mask_batch})
                    train_fruit_loss = session.run(fruit_error, feed_dict={obs_t_ph:obs_t_batch,
                                                                         act_t_ph:act_t_batch,
                                                                         rew_fruit_t_ph:rew_fruit_t_batch,
                                                                         obs_tp1_ph:obs_tp1_batch,
                                                                         done_mask_ph:done_mask_batch})
                    train_eat_loss = session.run(eat_error, feed_dict={obs_t_ph:obs_t_batch,
                                                                         act_t_ph:act_t_batch,
                                                                         rew_eat_t_ph:rew_eat_t_batch,
                                                                         obs_tp1_ph:obs_tp1_batch,
                                                                         done_mask_ph:done_mask_batch})

                    train_loss = (train_food_loss + train_avoid_loss + train_fruit_loss + train_eat_loss)/4.
                    print("\n \
                           Food loss: {}\n \
                           Avoid loss: {}\n \
                           Fruit loss: {}\n \
                           Eat loss: {}".format(train_food_loss,
                                                train_avoid_loss,
                                                train_fruit_loss,
                                                train_eat_loss))
                    print("Average loss at iteration {} is: {}".format(t, train_loss))
                num_param_updates += 1

                #####




            if args.count_lives:
                player.check_state()
            if player.done:
                break

        if player.done:
            player.eps_len = 0
            player.current_life = 0
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _ = player.model(
                (Variable(player.state.unsqueeze(0)), (player.hx, player.cx)))
            R = value.data

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + \
                args.gamma * player.values[i + 1].data - \
                player.values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                player.log_probs[i] * Variable(gae) - \
                0.01 * player.entropies[i]

        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(player.model.parameters(), 40)
        ensure_shared_grads(player.model, shared_model)
        optimizer.step()
        player.clear_actions()
