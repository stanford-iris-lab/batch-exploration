import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
weight = 1


"""
replay buffer dim:
    max_buffer_len, 5 X traj_len = 10
"""
def _count(buffer_id, args):
    threshold = args.threshold
    f = h5py.File(buffer_id + '.hdf5', 'r')
    qpos_init = f['sim']['states'][0, 0, :]
    qpos = f['sim']['states'][:, :, :]
    distractors_n = []
    if not (args.door or args.drawer):
        for i in range(3):
            moved = np.linalg.norm(qpos[:,:,(i+1)*3:(i+2)*3] - qpos_init[(i+1)*3:(i+2)*3], axis=-1)
            moved = moved.reshape(100, -1)
            moved_n = sum(np.max(moved, -1) > args.threshold)
            if args.goal_block == i:
                target = moved_n
            else:
                distractors_n.append(moved_n)
    else:
        moved = np.linalg.norm(qpos[:,:,-1] - qpos_init[-1], axis=-1)
        moved = moved.reshape(100, -1)
        moved_n = sum(np.max(moved, -1) > args.threshold)
        target = moved_n
        qpos = qpos[:,:,3:-1]
        qpos_init = qpos_init[3:-1]
        distractors = np.split(qpos, qpos.shape[2] // 3, axis=-1)
        distractor_init = np.split(qpos_init, len(qpos_init) // 3)
        for d in range(len(distractors)):
            moved = np.linalg.norm(distractors[d] - distractor_init[d], axis=-1)
            moved = moved.reshape(100, -1)
            moved_n = sum(np.max(moved, -1) > args.threshold)
            distractors_n.append(moved_n)

    print("Target Num : ", target, " | Distractor Num: ", distractors_n)
    distractors_n = np.array(distractors_n)
    return distractors_n/100.,target/100.


def count(args):
    distractors = np.zeros((args.buffer_num, 2), dtype=float)
    if args.door == 3 or args.door == 5: # 3 or 5 distractor towers + 1 target
        distractors = np.zeros((args.buffer_num, args.door), dtype=float)
    elif args.drawer: # 6 distractors blocks + 1 target
        distractors = np.zeros((args.buffer_num, 6), dtype=float)
    target = np.zeros((args.buffer_num,), dtype=float)
    savedir = args.pwd
    for buffer_id in range(args.buffer_num):
        distractor, tar = _count(savedir + str(buffer_id), args)
        distractors[buffer_id, :] = distractor[:]
        target[buffer_id] = tar

    np.save(savedir + 'distractors.npy', np.array(distractors))
    np.save(savedir + 'target.npy', np.array(target))

    plt.title('Target v.s. Distractor Interaction')
    plt.plot(target, label='target', marker='o')
    for d in range(distractors.shape[1]):
        plt.plot(distractors[:, d], label='distractor {}'.format(d), linestyle='--', marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Interaction Rate')
    plt.legend()
    plt.savefig(savedir + 'interaction.png')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pwd", type=str, default="exps/")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--buffer_num", type=int, default=21) 
    parser.add_argument("--door", type=int, default=0) # only works with either 3 or 5 (number of distractors) 
    parser.add_argument("--goal_block", type=int, default=1) # goal block between 0, 1, 2
    parser.add_argument("--drawer", type=bool, default=False)
    
    args = parser.parse_args()
    print(args.pwd)
    count(args)
