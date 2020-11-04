from PIL import Image
import torch
import numpy as np
import imageio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pix_augment(im):
    # im is (batch_sz, 3, 64, 64) as passed in
    im = im.to('cpu').detach().numpy()
    left_b = np.repeat(im[:,:,:1,:], 4, axis=2)
    right_b = np.repeat(im[:,:,-1:,:], 4, axis=2)
    aug_im = np.concatenate((left_b, im, right_b), axis=2)
    top_b = np.repeat(aug_im[:,:,:,:1], 4, axis=3)
    bottom_b = np.repeat(aug_im[:,:,:,-1:], 4, axis=3)
    aug_im = np.concatenate((top_b, aug_im, bottom_b), axis=3)
    # choose random (x,y) as top left corner of the crop
    rands = np.random.randint(0, 72-64, size=2)
    aug_im = aug_im[:,:, rands[1]:rands[1]+64, rands[0]:rands[0]+64]
    aug_im = torch.tensor(aug_im).to(device)
    return aug_im


'''Generates & Saves goal imgs in filepath'''
def get_goal_imgs(args, env, filepath):
    import os
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    
    goals = []
    def save_img(goal_num, im):
        img = Image.fromarray(im.astype(np.uint8))
        img.save(filepath + '/sample_goal' + str(goal_num) + '.png')

    if args.robot:
        for i in range(args.num_goal_images):
            im = env.get_goal()
            save_img(i, im)
            goals.append(im)
    elif args.mult_block: # trying to get interaction from multiple blocks
        for i in range(args.num_goal_images//2):
            im = env.get_goal(block=2)
            save_img(i, im)
            goals.append(im)
        for j in range(args.num_goal_images//2, args.num_goal_images + 1):
            im = env.get_goal(block=1)
            save_img(j, im)
            goals.append(im)
    else:
        for i in range(args.num_goal_images):
            if args.door or args.drawer:
                im = env.get_goal(block=None)
            elif args.goal_block == 0:
                im = env.get_goal(block=0)
            elif args.goal_block == 1:
                im = env.get_goal(block=1)
            elif args.goal_block == 2:
                im = env.get_goal(block=2)
            save_img(i, im)
            goals.append(im)
    return goals


def get_obs(args, info):
    hand = np.array([info['hand_x'], info['hand_y'], info['hand_z']])
    block0 = np.array([info['block0_x'], info['block0_y'], info['block0_z']])
    block1 = np.array([info['block1_x'], info['block1_y'], info['block1_z']])
    block2 = np.array([info['block2_x'], info['block2_y'], info['block2_z']])
    init_low_dim = np.concatenate([hand, block0, block1, block2])
    if args.drawer:
        block3 = np.array([info['block3_x'], info['block3_y'], info['block3_z']])
        block4 = np.array([info['block4_x'], info['block4_y'], info['block4_z']])
        block5 = np.array([info['block5_x'], info['block5_y'], info['block5_z']])
        drawer = np.array([info['drawer']])
        init_low_dim = np.concatenate([hand, block0, block1, block2, block3, block4, block5, drawer])
    elif args.door == 5:
        block3 = np.array([info['block3_x'], info['block3_y'], info['block3_z']])
        block4 = np.array([info['block4_x'], info['block4_y'], info['block4_z']])
        door = np.array([info['door']])
        init_low_dim = np.concatenate([hand, block0, block1, block2, block3, block4, door])
    elif args.door: 
        door = np.array([info['door']])
        init_low_dim = np.concatenate([hand, block0, block1, block2, door])
    return init_low_dim


'''For training a reward classifier for downstream planning'''
def train_reward_classifier(args):
    data_root = args.log_dir
    max_pos = data_root + 'max_pos/'
    modelvar_pos = data_root + 'modelvar_pos/'
    neg = data_root + 'neg/'
    val_pos = data_root + 'val_pos/'
    val_neg = data_root + 'val_neg/'
    root_dir = args.root

    enc_dec = SimpleVAE(device, args.latent_dim, args.log_dir)
    enc_dec.load_state_dict(torch.load(root_dir + '/enc_dec/1000model.bin'))
    enc_dec.to(device)
    classifier = BinClassifier(args.latent_dim, args.log_dir + '/classifier', 0).to(device)
    c_optimizer = optim.Adam(list(classifier.params), lr=1e-3)
    
    pos_data = [] # np array of the pos frames
    import glob
    for image_path in glob.glob(modelvar_pos + '*.png'): #max_pos
        image = imageio.imread(image_path)
        pos_data.append(image)
    neg_data = [] # np array of the pos frames
    for image_path in glob.glob(neg + '*.png'):
        image = imageio.imread(image_path)
        neg_data.append(image)
    
    val_pos_data = []
    for image_path in glob.glob(val_pos + '*.png'):
        image = imageio.imread(image_path)
        val_pos_data.append(image)
    val_neg_data = []
    for image_path in glob.glob(val_neg + '*.png'):
        image = imageio.imread(image_path)
        val_neg_data.append(image)
    
    pos_data = np.array(pos_data)
    neg_data = np.array(neg_data)
    val_pos_data = np.array(val_pos_data)
    val_pos_data = torch.tensor(val_pos_data).float().to(device).permute(0, 3, 1, 2) 
    val_neg_data = np.array(val_neg_data)
    val_neg_data = torch.tensor(val_neg_data).float().to(device).permute(0, 3, 1, 2) 
    _, _, _, _, pos_val_z = enc_dec.forward(val_pos_data.unsqueeze(1))
    _, _, _, _, neg_val_z = enc_dec.forward(val_neg_data.unsqueeze(1))
        
    losses = []
    val_acc = []
    for epoch in range(20):
        start = time.time()
        for i in range(100): #10 iters per epoch
            # Get batch
            pos = np.random.randint(0, pos_data.shape[0], args.batch_sz) 
            pos_obs = pos_data[pos]
            pos_obs = torch.tensor(pos_obs).float().to(device).permute(0, 3, 1, 2) 
            _, _, _, _, pos_obs_z = enc_dec.forward(pos_obs.unsqueeze(1))
            neg = np.random.randint(0, neg_data.shape[0], args.batch_sz) 
            neg_obs = neg_data[pos]
            neg_obs = torch.tensor(neg_obs).float().to(device).permute(0, 3, 1, 2) 
            _, _, _, _, neg_obs_z = enc_dec.forward(neg_obs.unsqueeze(1))
            
            # Train model
            score_path = None
            if args.logging > 0 and epoch % 10 == 0:
                score_path = args.log_dir + '/classifier_scores/' + str(epoch)
                if not os.path.exists(score_path):
                    os.makedirs(score_path)
            auxillary_loss = train_classifiers([classifier], enc_dec, neg_obs, pos_obs, c_optimizer, args.batch_sz, score_path)
            if i % 10 == 0:
                print("Iter", i, "loss", auxillary_loss)
        losses.append(auxillary_loss)
            
        end = time.time()
        print("===== EPOCH {} FINISHED IN {}s =====".format(epoch, end - start), "Classifier loss:", auxillary_loss)
        np.savetxt(args.log_dir + '/class_losses.txt', np.array(losses), fmt='%f')
        if epoch % 5 == 0:
            torch.save(classifier.state_dict(), args.log_dir + '/classifier' + '/{}model.bin'.format(epoch))
            
        # Eval on val set
        y_pos = classifier.forward(pos_val_z, calc_loss=False).squeeze().squeeze().cpu()
        acc_pos = float(((y_pos[:] > 0.5) == torch.ones(y_pos.shape[0])[:]).sum()) / y_pos.shape[0]
        y_neg = classifier.forward(neg_val_z, calc_loss=False).squeeze().squeeze().cpu()
        acc_neg = float(((y_neg[:] < 0.5) == torch.ones(y_neg.shape[0])[:]).sum()) / y_neg.shape[0]
        val_acc.append((acc_pos + acc_neg)/2)
        np.savetxt(args.log_dir + '/val_acc.txt', np.array(val_acc), fmt='%f')