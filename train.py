import torch
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
from utils import calculate_gp, lambda_gp, lambda_adv, make_grid, make_grid2, make_curve, write_loss, save_curve
import torch.nn as nn
import time
import os
import matplotlib.pyplot as plt

def train(G, D, start_epochs, end_epochs, batch_size, train_dataset, optimizer_G, optimizer_D, valid_criterion, label_criterion,
          latent_dim, classes_dim, output_path, device):

    schedual_D = None
    schedual_G = None
    circle = len(train_dataset)
    total = circle*batch_size
    G.train()
    D.train()
    iter = start_epochs*circle
    '''
    schedual_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.95)
    schedual_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.95)
    '''
    '''
    schedual_G = torch.optim.lr_scheduler.StepLR(optimizer_G, gamma=0.1, step_size=10)
    schedual_D = torch.optim.lr_scheduler.StepLR(optimizer_D, gamma=0.1, step_size=10)
    '''
    D_losses = []
    G_losses = []

    for epoch in range(start_epochs, end_epochs):

        start_time = time.time()

        for i, batch in enumerate(train_dataset):
          
            real_imgs = batch['image'].to(device)
            real_labels = batch['label'].to(device)
            batch_size = real_imgs.shape[0]
            valid = Variable(torch.FloatTensor(batch_size).fill_(1.0).to(device))
            fake = Variable(torch.FloatTensor(batch_size).fill_(0.0).to(device))

            '''
            Train D
            '''
            optimizer_D.zero_grad()

            #real image
            pred_valid_real, pred_label_real = D(real_imgs)

            errD_valid_real = valid_criterion(torch.squeeze(pred_valid_real), valid)
            errD_label_real = label_criterion(pred_label_real, real_labels)

            errD_real = errD_valid_real * lambda_adv + errD_label_real
            errD_real.backward()

            #fake image
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim-classes_dim))).to(device))
            gen_labels = Variable(torch.FloatTensor(np.random.randint(0, 2, (batch_size, classes_dim))).to(device))
            input = torch.cat((z, gen_labels.clone()), 1)
            fake_imgs = G(input)

            pred_valid_fake, pred_label_fake = D(fake_imgs.detach())

            errD_valid_fake = valid_criterion(torch.squeeze(pred_valid_fake), fake)
            errD_label_fake = label_criterion(pred_label_fake, gen_labels)

            errD_fake = errD_valid_fake * lambda_adv + errD_label_fake
            errD_fake.backward()

            #gp
            alpha = torch.FloatTensor(np.random.random(size=real_imgs.shape)).to(device)
            t = torch.rand(real_imgs.size()).to(device)
            interpolates = alpha * real_imgs.data + (1 - alpha) * (real_imgs.data + 0.5 * real_imgs.data.std() * t)
            interpolates = Variable(interpolates, requires_grad=True)
            d_interpolates, _ = D(interpolates)
            fake = torch.ones(d_interpolates.size()).to(device)
            gp = autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            errD_gp = lambda_gp * ((gp.norm(2, dim=1) - 1) ** 2).mean()
            errD_gp.backward()

            errD = errD_real + errD_fake + errD_gp

            optimizer_D.step()

            '''
            Train G
            '''
            optimizer_G.zero_grad()

            pred_valid_fake, pred_label_fake = D(fake_imgs)
            errG_valid = valid_criterion(torch.squeeze(pred_valid_fake), valid)
            errG_label = label_criterion(pred_label_fake, gen_labels)
            errG = errG_label + errG_valid * lambda_adv

            errG.backward()

            optimizer_G.step()

            iter += 1
            if iter == 50000:
                schedual_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.1)
                schedual_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.1)
                '''
                schedual_G = torch.optim.lr_scheduler.StepLR(optimizer_G, gamma=0.1, step_size=5)
                schedual_D = torch.optim.lr_scheduler.StepLR(optimizer_D, gamma=0.1, step_size=5)
                '''

            if iter % 50 == 0:
                #make_grid2(8, real_imgs, output_path, iter, 'real')
                make_grid2(8, fake_imgs, output_path, iter, 'fake')

            print('errG: ' + str(errG/batch_size) + ' errD: ' + str(errD/batch_size))
            write_loss(errD / batch_size, errG / batch_size, errD_fake/batch_size, errD_real/batch_size, errD_gp/batch_size, output_path)

            print('epoch ' + str(epoch) + ' process: ' + str(((i+1)*batch_size*1.0*100)/total) + '% ' + 'lr: ' + str(optimizer_G.param_groups[0]['lr']) + '(' + str(iter) + ')')

        end_time = time.time()
        cost_time = end_time - start_time
        print('epoch ' + str(epoch) + ' cost: ' + str(cost_time))
        save_curve(start_epochs, epoch, output_path)

        if schedual_D is not None:
            schedual_G.step()
            schedual_D.step()

        if (epoch+1) % 2 == 0:
            torch.save(G.state_dict(), os.path.join(output_path, 'G(1gp_no_gen_bn)_epoch' + str(epoch+1) + '.pth'))
            torch.save(D.state_dict(), os.path.join(output_path, 'D(1gp_no_gen_bn)_epoch' + str(epoch+1) + '.pth'))
