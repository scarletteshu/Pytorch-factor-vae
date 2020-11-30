import os
import random
import configs as cfg
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from fdataLoad import Dataloader

import torch
from torch import Tensor
import torch.nn.functional as F
from fcvae import FactorVAE, Discriminator


class RunModel:
    def __init__(self,):
        self.dl = Dataloader()
        self.dl.datasplit()

        # coefficient
        self.MAXEPOCH = cfg.params['EPOCH']
        self.num_iter = 0
        self.epoch = 0
        self.batch_size = cfg.params['batch_size']
        self.sch_gamma = cfg.params['sch_gamma']
        self.lr = cfg.params['lr']

        # model
        self.curr_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model = FactorVAE(1, 10).to(self.curr_device)
        self.D = Discriminator(10).to(self.curr_device)

        self.optimizer_vae = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        self.scheduler_vae = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_vae, gamma=self.sch_gamma)
        self.scheduler_D = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_D, gamma=self.sch_gamma)

    # Update the VAE
    def vaeLoss(self, recons, target, mu, log_var, D_z):
        # kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, target, reduction='sum').div(self.batch_size)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        vae_tc_loss = (D_z[:, 0] - D_z[:, 1]).mean()

        loss = recons_loss + kld_loss + self.model.gamma * vae_tc_loss
        # print(f' recons: {recons_loss}, kld: {kld_loss}, VAE_TC_loss: {vae_tc_loss}')
        return loss, recons_loss, -kld_loss, vae_tc_loss

    # update discriminator
    def discrimilatorLoss(self, z, D_z, **kwargs):
        true_labels = torch.ones(z.size(0), dtype=torch.long, requires_grad=False).to(z.device)
        false_labels = torch.zeros(z.size(0), dtype=torch.long, requires_grad=False).to(z.device)

        z_perm = self.model.permute_latent(z).detach()  # Detach so that VAE is not trained again

        D_z_perm    = self.D(z_perm)
        D_tc_loss = 0.5 * (F.cross_entropy(D_z, false_labels) + F.cross_entropy(D_z_perm, true_labels))

        return D_tc_loss

    def recordLoss(self, e, loss, recons_loss, kld_loss, vae_tc_loss, D_tc_loss, states="train"):
        f_tr = open(cfg.params["results_path"] + states + "/loss/" + states[:2] + "_loss_epoch_" + str(e) + ".pkl", "ab+")

        pkl.dump(np.array([
            loss,
            recons_loss,
            kld_loss,
            vae_tc_loss,
            D_tc_loss
        ]), f_tr)

    def recordResults(self, e: int, x: Tensor, type: str, states="train"):
        # exp: "./results/train/latents/tr_recons_epoch_0.pkl"
        f = open(cfg.params['results_path'] + states + "/" + type + "/" + states[:2] + "_" + type + "_epoch_" + str(e) + ".pkl", "ab+")
        pkl.dump(x.squeeze(1).to('cpu').detach().numpy(), f)
        f.close()

    def train(self):
        # checkpoint loading
        if os.path.isfile(cfg.params['models_path'] + "fvae.pth.tar"):
            print("model loading......")
            # only for inference: self.model.load_state_dict(torch.load("model.pth.tar"))
            checkpoint = torch.load(cfg.params['models_path'] + "fvae.pth.tar")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.optimizer_vae.load_state_dict(checkpoint['optimizer_vae_state_dict'])
            self.scheduler_vae.load_state_dict(checkpoint['scheduler_vae_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
            self.epoch = checkpoint['epoch']

        for e in range(self.epoch, self.MAXEPOCH):
            # for k cross validation
            k = e % 10
            tr_dataloader = self.dl.train_dataloader(k)
            # print("lr:{}".format(self.optim.param_groups[0]['lr']))

            self.model.train()
            for i, (target_img1, target_img2) in enumerate(tr_dataloader):
                self.num_iter += 1

                # update vae
                target_img1 = target_img1.to(self.curr_device)
                recons, mu, log_var, z = self.model(target_img1)  # recons_img, mu, log_var, latent z

                if i % 100 == 0:
                    self.recordResults(e, recons, "recons")
                    self.recordResults(e, target_img1, "origins")
                    self.recordResults(e, z, "latents")

                #print(recons.shape, target_img1.shape)
                D_z = self.D(z)
                loss, recons_loss, kld, vae_tc_loss = self.vaeLoss(recons, target_img1, mu, log_var, D_z)

                self.optimizer_vae.zero_grad()
                loss.backward(retain_graph=True)

                # update discriminator
                target_img2 = target_img2.to(self.curr_device)
                _, _, _, z_prime = self.model(target_img2)
                loss_D = self.discrimilatorLoss(z_prime, D_z)

                self.optimizer_D.zero_grad()
                loss_D.backward()
                self.optimizer_vae.step()
                self.optimizer_D.step()

                print("epoch {}, iter {}\n".format(e, self.num_iter),
                      "loss:{}, recons_loss:{}, kld:{}, vae_tc_loss{}, ".format(loss.item(),
                                                                                recons_loss.item(),
                                                                                kld.item(),
                                                                                vae_tc_loss.item()),
                      "D_tc_loss:{}".format(loss_D.item()))

                if i % 50 == 0:
                    self.recordLoss(e, loss.item(),
                                       recons_loss.item(),
                                       kld.item(),
                                       vae_tc_loss.item(),
                                       loss_D.item())

            torch.save({
                'epoch': e+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_vae_state_dict': self.optimizer_vae.state_dict(),
                'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                'scheduler_vae_state_dict': self.scheduler_vae.state_dict(),
                'scheduler_D_state_dict': self.scheduler_D.state_dict(),
                'D_state_dict': self.D.state_dict()
            }, cfg.params['models_path'] + "fvae.pth.tar")

            self.model.eval()
            # validate after every epoch
            self.validate(e, k)
            # decay for every 10 epoch
            if e % 10 == 0:
                self.scheduler_vae.step()
                self.scheduler_D.step()

    def validate(self, e: int, k: int):
        val_loss = []
        val_recons_loss = []
        val_kld_loss = []
        val_vae_tc_loss = []
        val_D_tc_loss = []
        val_dataloader = self.dl.val_dataloader(k)

        with torch.no_grad():
            for i, (val_img1, val_img2) in enumerate(val_dataloader):
                val_img1 = val_img1.to(self.curr_device)
                recons, mu, log_var, z = self.model(val_img1)
                D_z = self.D(z)
                loss, recons_loss, kld, vae_tc_loss = self.vaeLoss(recons, val_img1, mu, log_var, D_z)

                val_img2 = val_img2.to(self.curr_device)
                _, _, _, z_prime = self.model(val_img2)
                loss_D = self.discrimilatorLoss(z_prime, D_z)

                if i % 200 == 0:
                    self.recordResults(e, recons, "recons", "val")
                    self.recordResults(e, val_img1, "origins", "val")
                    #self.recordResults(e, z, "latents", "val")

                val_loss.append(loss.item())
                val_recons_loss.append(recons_loss.item())
                val_kld_loss.append(kld.item())
                val_vae_tc_loss.append(vae_tc_loss.item())
                val_D_tc_loss.append(loss_D.item())

            avg_loss = sum(val_loss)/len(val_loss)
            avg_recons_loss = sum(val_recons_loss)/len(val_recons_loss)
            avg_kld_loss = sum(val_kld_loss)/len(val_kld_loss)
            avg_vae_tc_loss = sum(val_vae_tc_loss)/len(val_vae_tc_loss)
            avg_D_tc_loss = sum(val_D_tc_loss)/len(val_D_tc_loss)

            print("-----------------validating------------------")
            print("epoch{}\tavg_loss:{}, avg_recons_Loss:{}, avg_KLD:{}, avg_vae_tc_loss:{}\n".format(e, avg_loss, avg_recons_loss, avg_kld_loss, avg_vae_tc_loss),
                  "avg_D_tc_loss:{}".format(avg_D_tc_loss))
            print("---------------------------------------------")

            self.recordLoss(e, avg_loss, avg_recons_loss, avg_kld_loss, avg_vae_tc_loss, avg_D_tc_loss, states="val")

    """
    def sample_images(self):

        test_input, test_label = next(iter(self.dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)

        vutils.save_image(recons.data, f"{self.log_dir}/recons_{self.cur_epoch}".png,normalize=True, nrow=12)

        #vutils.save_image(recons.data,
        #                  f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                  f"recons_{self.logger.name}_{self.current_epoch}.png",
        #                  normalize=True,
        #                  nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144, self.curr_device, labels=test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.log_dir}/recons_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except:
            pass

        del test_input, recons  # , samples
        """

    def save_traverse(self, sample, fixed_name, vary_dim, val):
        save_dir = './results/plot/latent_trav/' + fixed_name + '/' + str(vary_dim) + '_' + str(val) + '.jpg'
        sample = sample.to('cpu').numpy()
        plt.imsave(save_dir, sample, cmap='gray')

    def latent_traverse(self, limit=3, inter=2 / 3, loc=-1):
        if os.path.isfile("fvae.pth.tar"):
            print("model loading......")
            # only for inference: self.model.load_state_dict(torch.load("model.pth.tar"))
            checkpoint = torch.load("fvae.pth.tar")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        tr_dataloader = self.dl.train_dataloader(7)
        model.eval()
        # print("lr:{}".format(self.optim.param_groups[0]['lr']))
        for i, (target_img1, _ ) in enumerate(tr_dataloader):
            self.model.eval()

            target_img1 = target_img1.to(self.curr_device)
            # [mu, log_var] = self.model.encode(target_img1)
            # random_img_z = self.model.reparameterize(mu, log_var)
            re, _, _, _ = self.model(target_img1)
            print(re.shape)
            plt.imsave("random.jpg", target_img1.cpu().data.squeeze(), cmap='gray')
            plt.imsave("rand_re.jpg", re.cpu().data.squeeze(), cmap='gray')

            interpolation = torch.arange(-limit, limit + 0.1, inter)
            # print(interpolation)

            z_ori = random_img_z

                # samples.append(sample)
                # samples = []
            for row in range(6):
                if loc != -1 and row != loc:
                        continue
                z = z_ori.clone()
                for val in interpolation:  # 一维改变
                    z[:, row] = val
                    print(z)
                    sample = self.model.decode(z).data.squeeze()
                    print(sample.shape)
                    self.save_traverse(sample, "random_img", row, val)
                    # samples.append(sample)
            break
                # samples = torch.cat(samples, dim=0).cpu()

        # latents = torch.from_numpy(sample_latent(latents_sizes).astype('float32'))
        """
        dsets_len = self.dl.dataset.__len__()
        rand_idx = random.randint(1, dsets_len - 1)

        random_img, _ = self.dl.dataset.__getitem__(rand_idx)
        random_img = random_img.unsqueeze(0)
        random_img = random_img.to(self.curr_device)
        [mu, log_var] = self.model.encode(random_img)
        random_img_z = self.model.reparameterize(mu, log_var)
        re, _,_,_ = self.model(random_img)
        plt.imsave("rand_re.jpg", re.cpu().data.squeeze(), cmap='gray')

        fixed_idx1 = 87040  # square
        fixed_idx2 = 332800  # ellipse
        # fixed_idx3 = 578560 # heart

        fixed_img1, _ = self.dl.dataset.__getitem__(fixed_idx1)
        fixed_img1 = fixed_img1.unsqueeze(0).to(self.curr_device)
        print(fixed_img1.shape)
        [mu, log_var] = self.model.encode(fixed_img1)
        fixed_img_z1 = self.model.reparameterize(mu, log_var)

        fixed_img2, _ = self.dl.dataset.__getitem__(fixed_idx2)
        fixed_img2 = fixed_img2.unsqueeze(0).to(self.curr_device)
        [mu, log_var] = self.model.encode(fixed_img2)

        fixed_img_z2 = self.model.reparameterize(mu, log_var)
        
        fixed_img3 = self.dl.dataset.__getitem__(fixed_idx3)
        fixed_img3 = Variable(cuda(fixed_img3, self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]
        """
        """

        Z = {'fixed_square': fixed_img_z1, 'fixed_ellipse': fixed_img_z2,
             'random_img': random_img_z}

        interpolation = torch.arange(-limit, limit + 0.1, inter)
        # print(interpolation)

        for key in Z.keys():
            z_ori = Z[key]


            # samples.append(sample)
            # samples = []
            for row in range(6):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:  # 一维改变
                    z[:, row] = val
                    print(z)
                    sample = self.model.decode(z).data.squeeze()
                    print(sample.shape)
                    self.save_traverse(sample, key, row, val)
                    # samples.append(sample)

            # samples = torch.cat(samples, dim=0).cpu()
        """

if __name__ == "__main__":
    run = RunModel()
    print(run.model)
    run.train()
    #run.latent_traverse()
