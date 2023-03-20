import torch
import torch.optim as optim
from datasets import VoxelDataset
from torch.utils.data import DataLoader
import random
import numpy as np
import time
from tqdm import tqdm
from model.autoencoder_introspective import Autoencoder, amcm, LATENT_CODE_SIZE

'''
This Autoencoder model is adapted from marian42's autoencoder "https://github.com/marian42/shapegan", following the
idea of IntroVAE "https://github.com/hhb072/IntroVAE"
'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(10)
torch.manual_seed(10)
continue_learning = False
batch_size = 128
voxel_size = 32
show_viewer = True
m_plus = 1000
weight_neg = 0.1  #
weight_rec = 10   #beta
weight_kl = 0.25   #alpha
lr_e, lr_g = 3e-6, 3e-6
epoch_save_freq = 20
view_freq = 10

dataset = VoxelDataset.from_split('data/montreal/voxels_{:d}/{{:s}}.npy'.format(voxel_size), 'data/montreal/train_final.txt')
data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
testset = VoxelDataset.from_split('data/montreal/voxels_{:d}/{{:s}}.npy'.format(voxel_size), 'data/montreal/test_final.txt')
test_loader = DataLoader(testset, shuffle=True, batch_size=batch_size, num_workers=0)
print("trainset:  ", len(dataset), "\ttestset:  ", len(testset))

autoencoder = Autoencoder(vox_res = voxel_size, is_variational=True)

if continue_learning:
    autoencoder.load()

optimizerE = optim.Adam(autoencoder.encoder.parameters(), lr=lr_e)
optimizerG = optim.Adam(autoencoder.decoder.parameters(), lr=lr_g)

# scheduler_e = optim.lr_scheduler.StepLR(optimizerE, step_size=50, gamma=0.7681904)
# scheduler_g = optim.lr_scheduler.StepLR(optimizerG, step_size=50, gamma=0.7681904)
# scheduler_e = optim.lr_scheduler.StepLR(optimizerE, step_size=50, gamma=0.9198813466998584,verbose=True)
# scheduler_g = optim.lr_scheduler.StepLR(optimizerG, step_size=50, gamma=0.9198813466998584)


if show_viewer:
    from rendering import MeshRenderer
    viewer = MeshRenderer()

log_file_name = "plots/IntroVAE_final_param_training_{}vox_{}comp_RENDERINGTEST.csv".format(voxel_size,amcm)
log_file = open(log_file_name, "a" if continue_learning else "w")
first_epoch = 0

def voxel_difference(input, target):
    wrong_signs = (input * target) < 0
    return torch.sum(wrong_signs).item() / wrong_signs.nelement()

if continue_learning:
    log_file_contents = open(log_file_name, 'r').readlines()
    first_epoch = len(log_file_contents)

print("first epoch: ", first_epoch)


def kld_loss(mean, log_variance):
    return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp()) / mean.nelement()


def get_reconstruction_loss(input, target, size_average = False):
    difference = input - target
    wrong_signs = target < 0
    difference[wrong_signs] *= voxel_size
    if size_average:
        return torch.mean(torch.abs(difference))
    else:
        return torch.sum(torch.abs(difference))

def train(max_epoch):
    autoencoder.train()
    for epoch in range(first_epoch, max_epoch):
        epoch_start_time = time.time()
        batch_index = 0

        loss_rec_history, test_loss_rec_history = [], []
        lossE_real_kl_history, lossE_rec_kl_history, lossE_fake_kl_history = [], [], []
        lossG_rec_kl_history, lossG_fake_kl_history = [], []

        for batch in tqdm(data_loader, desc='Epoch {:d}'.format(epoch)):
            noise = torch.zeros(batch_size, LATENT_CODE_SIZE).normal_(0, 1).to(device)   # noise is batch random z
            real = batch.to(device) # real is batch voxels

            # =========== Update E ================
            fake = autoencoder.decode(noise)   # random reconstruction from decoder
            real_mu, real_logvar, z, rec = autoencoder(real)
            rec_z, rec_mu, rec_logvar = autoencoder.encode(rec.detach(),return_mean_and_log_variance= True)
            fake_z, fake_mu, fake_logvar = autoencoder.encode(fake.detach(),return_mean_and_log_variance= True)

            loss_rec = get_reconstruction_loss(rec, real, True)

            lossE_real_kl = kld_loss(real_mu, real_logvar).mean()
            lossE_rec_kl = kld_loss(rec_mu, rec_logvar).mean()
            lossE_fake_kl = kld_loss(fake_mu, fake_logvar).mean()

            loss_margin = lossE_real_kl + \
                          (torch.nn.functional.relu(m_plus - lossE_rec_kl) + \
                           torch.nn.functional.relu(m_plus - lossE_fake_kl)) * 0.5 * weight_neg
            lossE = loss_rec * weight_rec + loss_margin * weight_kl

            optimizerG.zero_grad()
            optimizerE.zero_grad()
            lossE.backward(retain_graph=True)
            optimizerE.step()

            # ========= Update G ==================
            rec_z, rec_mu, rec_logvar = autoencoder.encode(rec.detach(),return_mean_and_log_variance= True)
            fake_z, fake_mu, fake_logvar = autoencoder.encode(fake.detach(),return_mean_and_log_variance= True)

            lossG_rec_kl = kld_loss(rec_mu, rec_logvar).mean()
            lossG_fake_kl = kld_loss(fake_mu, fake_logvar).mean()
            lossG = (lossG_rec_kl + lossG_fake_kl) * 0.5 * weight_kl
            lossG.backward()
            optimizerG.step()

            # ========= Learning Criteria ==================
            loss_rec_history.append(loss_rec.item())
            lossE_real_kl_history.append(lossE_real_kl.item())
            lossE_rec_kl_history.append(lossE_rec_kl.item())
            lossE_fake_kl_history.append(lossE_fake_kl.item())
            lossG_rec_kl_history.append(lossG_rec_kl.item())
            lossG_fake_kl_history.append(lossG_fake_kl.item())

            batch_index+=1

        for test_batch in tqdm(test_loader, desc='Epoch {:d}'.format(epoch)):
            test_real = test_batch.to(device)  # real is batch voxels
            # =========== TEST E ================
            test_real_mu, test_real_logvar, test_z, test_rec = autoencoder(test_real)
            test_loss_rec = get_reconstruction_loss(test_rec, test_real, True)
            test_loss_rec_history.append(test_loss_rec.item())

        tqdm.write('Epoch {:d} ({:.1f}s), Train Rec / Test Rec: {:.6f} / {:.6f}, Kl_E: {:.4f}, {:.4f}, {:.4f}, Kl_G: {:.4f}, {:.4f}'.format(
            epoch,
            time.time() - epoch_start_time,
            np.array(loss_rec_history).mean(),
            np.array(test_loss_rec_history).mean(),
            np.array(lossE_real_kl_history).mean(),np.array(lossE_rec_kl_history).mean(),np.array(lossE_fake_kl_history).mean(),
            np.array(lossG_rec_kl_history).mean(),np.array(lossG_fake_kl_history).mean()))

        log_file.write("{:d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
            epoch,
            np.array(loss_rec_history).mean(),
            np.array(test_loss_rec_history).mean(),
            np.array(lossE_real_kl_history).mean(), np.array(lossE_rec_kl_history).mean(),np.array(lossE_fake_kl_history).mean(),
            np.array(lossG_rec_kl_history).mean(),np.array(lossG_fake_kl_history).mean()))
        log_file.flush()

        autoencoder.save()
        if epoch % epoch_save_freq == 0:
            autoencoder.save(epoch=epoch)

if __name__ == '__main__':
    train(1500)
    log_file.close()
