import numpy as np

def _door_goal(block_num=5):
    if block_num == 5:
        block_0_pos = [-0.15, 0.8, 0.075]
        block_1_pos = [-0.12, 0.6, 0.075]
        block_2_pos = [0.25, 0.4, 0.075]
        block_3_pos = [0.25, 0.6, 0.075]
        block_4_pos = [0.15, 0.6, 0.075]
        
        block_pos = np.concatenate([block_0_pos, block_1_pos, block_2_pos, block_3_pos, block_4_pos])
    
    elif block_num == 3:
        block_0_pos = [-0.15, 0.8, 0.075]
        block_1_pos = [-0.12, 0.6, 0.075]
        block_2_pos = [0.25, 0.4, 0.075]
        block_pos = np.concatenate([block_0_pos, block_1_pos, block_2_pos])
    return block_pos

def _drawer_goal():
    block_0_pos = [0.35, 0.3, 0.05]
    block_1_pos = [-0.12, 0.6, 0.05]
    block_2_pos = [0.2, 0.3, 0.05]
    block_3_pos = [-0.15, 0.4, 0.05]
    block_4_pos = [0.45, 0.6, 0.05]
    block_5_pos = [-0.2, 0.7, 0.05]
    blocks_pos = np.concatenate([block_0_pos, block_1_pos, block_2_pos, block_3_pos, block_4_pos, block_5_pos])
    return blocks_pos

def _block_goal(block_num=1, hard=False):
    if block_num == 0:
        block_1_pos = [0.0, 0.15, 0] + np.random.uniform(-0.02, 0.02, (3,))
        block_2_pos = [0.2, 0.15, 0] + np.random.uniform(-0.02, 0.02, (3,))
        block_0_pos = np.random.uniform(
                    (-0.2, 0.05, 0.0),  
                    (-0.0, 0.25, 0.20), 
                    size=(3,)) 
        if hard:
            block_1_pos = [-.1, .15, 0] + np.random.uniform(-0.02, 0.02, (3,))
            block_2_pos = [.2, -.1, 0] + np.random.uniform(-0.02, 0.02, (3,))
            block_0_pos = np.random.uniform((-.2, -0.2, 0.0), (.2, 0.2, 0.20), size=(3,))
          
        block_0_pos += np.random.uniform(-0.02, 0.02, (3,))
        gripper_pos = block_0_pos.copy()
        gripper_pos += np.random.uniform(-0.02, 0.02, (3,))
        gripper_pos[1] += 0.6 # need to adjust for middle of the table for the gripper being (0.0, 0.6)
        gripper_pos[2] = np.random.uniform(0.0, 0.20)
    
    elif block_num == 1:
        block_0_pos = [-0.1, 0.15, 0] + np.random.uniform(-0.02, 0.02, (3,)) 
        block_2_pos = [0.1, 0.15, 0] + np.random.uniform(-0.02, 0.02, (3,)) 
        block_1_pos = np.random.uniform(
                    (-0.1, 0.05, 0),
                    (0.1, 0.25, 0.20),
                    size=(3,))
        if hard:
            block_0_pos = [-.2, 0, 0] + np.random.uniform(-0.02, 0.02, (3,))
            block_2_pos = [.2, -.1, 0] + np.random.uniform(-0.02, 0.02, (3,)) 
            block_1_pos = np.random.uniform((-.2, -0.2, 0.0), (.2, 0.2, 0.20), size=(3,))
          
        block_1_pos += np.random.uniform(-0.02, 0.02, (3,))
        gripper_pos = block_1_pos.copy()
        gripper_pos += np.random.uniform(-0.02, 0.02, (3,))
        gripper_pos[1] += 0.6 # need to adjust for middle of the table for the gripper being (0.0, 0.6)
        gripper_pos[2] = np.random.uniform(0.0, 0.20)
    elif block_num == 2:
        block_1_pos = [0.0, 0.15, 0] + np.random.uniform(-0.02, 0.02, (3,))
        block_0_pos = [-0.1, 0.15, 0] + np.random.uniform(-0.02, 0.02, (3,))
        block_2_pos = np.random.uniform( 
                    (0.0, 0.1, 0.0),  
                    (0.2, 0.2, 0.20), 
                    size=(3,)) 
        if hard:
            block_0_pos = [-.2, 0, 0] + np.random.uniform(-0.02, 0.02, (3,))
            block_1_pos = [-.1, .15, 0] + np.random.uniform(-0.02, 0.02, (3,))
            block_2_pos = np.random.uniform((-.2, -0.2, 0.0), (.2, 0.2, 0.20), size=(3,))
          
        block_2_pos += np.random.uniform(-0.02, 0.02, (3,))
        gripper_pos = block_2_pos.copy()
        gripper_pos += np.random.uniform(-0.02, 0.02, (3,))
        gripper_pos[1] += 0.6
        gripper_pos[2] = np.random.uniform(0.0, 0.20)

    gripper_pos[2] = np.random.uniform(0.2, 0.30)
    # gripper_pos = [.0, 0.6, 0.3]
    goal_pos = np.concatenate((gripper_pos, block_0_pos, block_1_pos, block_2_pos))
    return goal_pos
