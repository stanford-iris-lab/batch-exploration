# Make a separate class just for logging (preds & losses)
import numpy as np

class Hist():
    def __init__(self, root):
        super().__init__()
        self.rankings_dir = root + '/rankings'
        self.loss_dir = root + '/losses'
        self.report_losses = {
                            'auxillary_loss': [],
                            'dynamics_loss': [],
                            'vae_loss': [],
                            }
        import os
        if not os.path.exists(self.rankings_dir):
            os.mkdir(self.rankings_dir)
        if not os.path.exists(self.loss_dir):
            os.mkdir(self.loss_dir)

    def save_losses(self, auxillary_loss, dynamics_loss, vae_loss):
        self.report_losses['auxillary_loss'].append(auxillary_loss)
        self.report_losses['dynamics_loss'].append(dynamics_loss)
        self.report_losses['vae_loss'].append(vae_loss)

    def save_losses_txt(self):
        np.savetxt(self.loss_dir + '/auxillary_losses.txt', np.array(self.report_losses['auxillary_loss']), fmt='%f')
        np.savetxt(self.loss_dir + '/dynamics_losses.txt', np.array(self.report_losses['dynamics_loss']), fmt='%f')
        np.savetxt(self.loss_dir + '/vae_losses.txt', np.array(self.report_losses['vae_loss']), fmt='%f')

    def save_preds(self):
        pass

    def save_rankings(self):
        pass
