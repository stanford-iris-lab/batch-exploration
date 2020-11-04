from utils.logging import *
import torch

SAMPLE_SZ = 1000
RESAMPLE_RATIO = .2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_random_action_sequence(env, traj_length, sample_sz=1000):
    act_dim = env.action_space.shape[0]
    acts = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=[sample_sz, traj_length, act_dim])
    return torch.tensor(acts).float().to(device)

def get_action_and_info(env, observation, traj_length, dynamics_model, enc_dec, actions=None):
    if actions is None:
        actions = get_random_action_sequence(env, traj_length, sample_sz=SAMPLE_SZ)
    with torch.no_grad():
        dynamics_model.eval()
        preds = None
        _, _, _, _, hidden = enc_dec.forward(observation.unsqueeze(1), reconstruct=False)
        obs_enc = hidden.repeat(SAMPLE_SZ, 1, 1)
        preds = dynamics_model.predict(obs_enc, actions)
        preds = preds.reshape(SAMPLE_SZ, -1, preds.shape[2])
    return actions, preds, hidden

def _compute_disagrmnt_reward(obs, actions, dynamics_models):
    preds_list = []
    obs = obs.repeat(SAMPLE_SZ, 1, 1)
    for model in dynamics_models:
        model.eval()
        preds = model.predict(obs, actions)
        preds = preds.reshape(SAMPLE_SZ, -1, preds.shape[2])
        preds_list.append(preds) 
    preds_list = torch.stack(preds_list)
    var_rewards = ((preds_list - preds_list.mean(0))**2).mean(axis=(0, -1))
    return var_rewards, preds_list.mean(0)

def _compute_classifier_reward(args, obs, classifiers):
    out = []
    sz = obs.shape[0]
    with torch.no_grad():
        for classifier in classifiers:
            classifier.eval()
            if args.instance_normalized:
                reward = classifier.forward(obs.reshape(-1, obs.shape[2]))
                out.append(reward.reshape(sz, -1, 1))
            else:
                out.append(classifier.forward(obs))
        out = torch.stack(out).squeeze()
        rew = out.max(0)[0]
        return rew

def _compute_smm_reward(args, obs, goal_vae, density_vae):
    h_s_z = density_vae.get_output_for(obs)
    pred_log_ratios = None
    p_star_loss = goal_vae.get_output_for(obs)
    pred_log_ratios = (args.p_star_weight * p_star_loss - h_s_z)
    pred_log_ratios = pred_log_ratios.squeeze(2)
    return pred_log_ratios

"""
Sampling-based planning. Then sort actions based on rewards.
"""
def plan_actions(args, env, ob, dynamics_model, enc_dec, classifiers=None, goal_vae=None, density_vae=None, dynamics_models=None):
    sample_actions, preds, obs_emb = get_action_and_info(env, ob, args.traj_length, dynamics_model, enc_dec)

    action_rewards = torch.zeros((preds.shape[:2]))
    if args.dynamics_var:
        action_rewards, preds = _compute_disagrmnt_reward(obs_emb, sample_actions, dynamics_models)
    elif args.smm:
        action_rewards = _compute_smm_reward(args, preds[:,:, :],  goal_vae, density_vae)
    elif args.use_classifiers is not None:
        action_rewards = _compute_classifier_reward(args, preds[:,:,:], classifiers)

    if args._resample:
        _, ind = torch.sort(action_rewards)
        _resample_sz = int(RESAMPLE_RATIO * SAMPLE_SZ)
        mask = ind < _resample_sz
        candidate_actions = sample_actions[mask]
        best_rewards = action_rewards[mask]
        mu1 = candidate_actions.mean(0)
        std1 = candidate_actions.std(0)
        sample_actions = mu1 + (torch.empty(sample_actions.shape).normal_(mean=0,std=1).to(device) * std1)
        sample_actions, preds = get_action_and_info(env, ob, args.traj_length, dynamics_model, enc_dec, actions=sample_actions)

        if args.dynamics_var:
            action_rewards, preds = _compute_disagrmnt_reward(ob, sample_actions, dynamics_models)
        elif args.smm:
            action_rewards = _compute_smm_reward(args, preds[:,:, :],  goal_vae, density_vae)
        elif args.use_classifiers is not None:
            action_rewards = _compute_classifier_reward(args, preds[:,:,:], classifiers)
    action_rewards = action_rewards.mean(-1) # average rewards from 10 steps
    action_rewards = ptu.get_numpy(action_rewards).tolist()
    sample_actions = ptu.get_numpy(sample_actions)
    preds = ptu.get_numpy(preds)
    best_rewards = np.array([x for _, x in sorted(zip(action_rewards, action_rewards), key=lambda y:y[0], reverse=True)])
    best_actions = np.array([x for _, x, in sorted(zip(action_rewards, sample_actions), key=lambda y:y[0], reverse=True)])
    ordered_preds = np.array([x for _, x, in sorted(zip(action_rewards, preds), key=lambda y:y[0], reverse=True)])
    return best_rewards, best_actions, ordered_preds

