import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard", default=True, action='store_true') # If True, initialize blocks to each corner
    parser.add_argument("--action_dim", type=int, default=5)
    parser.add_argument("--code_dim", type=int, default=12) #12 for low dim
    parser.add_argument("--net_size", type=int, default=256) 
    parser.add_argument("--replay_buffer_size", type=int, default=5000) # only for logging block interaction
    parser.add_argument("--memory_size", type=int, default=int(2e4)) # Size of memory to draw samples from

    parser.add_argument("--latent_dim", type=int, default=256) # Dimension of latent space if high dim, VAE_GOAL_DIM if low dim

    parser.add_argument("--root", type=str, default='exps/')
    parser.add_argument("--loss_log_freq", type=int, default=10)
    parser.add_argument("--env_log_freq", type=int, default=50) # 100 in terms of episode num
    parser.add_argument("--model_log_freq", type=int, default=100)
    parser.add_argument("--verbose", type=bool, default=1) # if logging everything from the environment

    parser.add_argument("--instance_normalized", default=False, action="store_true")
    parser.add_argument("--num_epochs", type=int, default=int(200e3))
    parser.add_argument("--traj_length", type=int, default=10)
    parser.add_argument("--num_traj_per_epoch", type=int, default=5) # 50 steps total per epoch/episode
    parser.add_argument("--update_freq", type=int, default=1) # update models every 10 10-step trajectories
    parser.add_argument("--batch_sz", type=int, default=32)

    parser.add_argument("--_resample", action='store_true', default=False) # refit on the best K trajectories
    parser.add_argument("--random_act_prob", default=0.1, type=float) # take random action while executing actions from the selected trajectory
    parser.add_argument("--goal_cond", type=bool, default=True)
    parser.add_argument("--p_star_weight", type=int, default=1000) # weight for log p*(s) term (only for smm)
    parser.add_argument("--beta", type=float, default=1e-3) # weight for vae kl loss relative to reconstruction (only for classifiers)
    parser.add_argument("--num_goal_images", type=int, default=100)

    # "both": use weighted mean + var
    # "mean": use only mean
    # "var": use only var
    # "max": use max(classifier) score w/ only 3 classifiers
    parser.add_argument("--use_classifiers", type=str, default=None) # If using ensemble of binary classifiers
    parser.add_argument("--num_classifiers", type=int, default=10) # Number of classifiers to use: if "max", then 3, otherwise 10
    parser.add_argument("--grad_steps_per_update", type=int, default=20) 
    parser.add_argument("--dynamics_var", action='store_true', default=False) # If using dynamics disagreement as a baseline
    parser.add_argument("--smm", action='store_true', default=False) # If using smm as a baseline
    parser.add_argument("--goal_block", type=int, default=2) # Choose target block btwn 0-2
    parser.add_argument("--door", type=int, default=0) # 1 for default door env, 3 for door with 3 distractor towers, and 5 for 5 distractors
    parser.add_argument("--add_noise", action='store_true', default=False) # If adding noise to the initial pos
    parser.add_argument("--mult_block", action='store_true', default=False) # If trying to target multiple blocks
    parser.add_argument("--drawer", action='store_true', default=False) # If using cluttered scene with target drawer

    # Logging types
    # 0: no logging 
    # 1: no viz env logging
    # 2: all logging
    parser.add_argument("--logging", default=2, type=int)
    parser.add_argument("--max_epoch", default=2101, type=int)

    parser.add_argument("--classifiers_grad_steps", default=1, type=int)
    parser.add_argument("--meanvar_weight", default=0.2, type=float)
    parser.add_argument("--classifiers_weight", default=10.0, type=float) # Factor to divide the classifier losses by

    parser.add_argument("--vae_no_goal", action='store_true', default=False)

    # Args for the log directory
    parser.add_argument("--date", type=str, default="06_28")
    parser.add_argument("--seed", type=int, default=0)

    ## ROBOT FLAGS
    parser.add_argument("--robot", action='store_true', default=False)
    parser.add_argument("--reload", type=str, default=None) 
    parser.add_argument("--reload_epoch", type=int, default=100)


    args = parser.parse_args()
    return args



def create_log_dir(args):
    if args.use_classifiers == "max":
        args.num_classifiers = 3
    
    # Build log directory
    logdir = args.root + args.date + '/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if args.use_classifiers is not None:
        logdir += args.use_classifiers + '_'
        if args.instance_normalized:
            logdir += 'in_' # classifier layers instance normalized
        else:
            logdir += 'sn_' # classifier weights spectral normalized
    elif args.dynamics_var:
        logdir += 'modelvar_'
    elif args.smm:
        logdir += 'smm_'
    
    if args.use_classifiers == "both":
        logdir += 'w' + str(args.meanvar_weight)
    
    if args.beta != 1e-3:
        logdir += '_beta' + str(args.beta) + '_'
    
    if args.vae_no_goal:
        logdir += '_vng_'
    
    logdir += '_seed' + str(args.seed) + '_'
    
    if args.door:
        logdir += 'door_' + str(args.door)
    elif args.drawer:
        logdir += 'drawer'
    else:
        logdir += 'block'
        logdir += str(args.goal_block)

    if args.robot:
        logdir += "_ROBOT"
             
    args.log_dir = logdir
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    return args
