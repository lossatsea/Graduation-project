import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.autograd as autograd
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import torchvision
import os

lambda_gp = 0.5
lambda_adv = 14

def weight_initialize(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

def FID(reals, fakes, illust2vec, col_num=2048):
    pic_num =  len(reals)
    real_features = np.empty((pic_num, col_num))
    fake_features = np.empty((pic_num, col_num))
    for i in range(pic_num):
        real_img = Image.open(reals[i])
        real_features[i] = illust2vec.extract_feature(real_img)
        fake_img = Image.open(fakes[i])
        fake_features[i] = illust2vec.extract_feature(fake_img)
    real_feature = np.mean(real_features, axis=0)
    fake_feature = np.mean(fake_features, axis=0)
    mu1 = real_feature
    mu2 = fake_feature
    c1 = np.cov(real_features, rowvar=False)
    c2 = np.cov(fake_features, rowvar=False)
    dis = (mu1 - mu2)**2 + np.trace(c1 + c2 - 2*np.sqrt(c1*c2))
    return dis

def calculate_gp(D, X, device):
    alpha = torch.FloatTensor(np.random.random(size=X.shape)).to(device)
    t = torch.rand(X.size()).to(device)
    interpolates = alpha*X.data + (1-alpha) * (X.data + 0.5*X.data.std() * t)
    #print(interpolates.size())
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

    return ((gp.norm(2, dim=1) - 1)**2).mean()

def show(img):
    npimg = img.numpy()
    plt.pause(0.05)
    return npimg

def denorm(x):
    # denormalize
    out = (x + 1) / 2
    return out.clamp(0, 1)

def make_grid(rows, imgs, output_path, epoch):
    list = []
    for i in range(imgs.shape[0]):
        list.append(imgs[i])
    path = os.path.join(output_path, 'sample_' + str(epoch) + '.jpg')
    torchvision.utils.save_image(denorm(torch.stack(list)), path, nrow=rows)
    print('ok')
    return denorm(torch.stack(list))

def make_grid2(rows, imgs, output_path, iter, type):
    path = os.path.join(output_path, type + '-sample_iter' + str(iter) + '.jpg')
    torchvision.utils.save_image(imgs, path, nrow=rows)

def make_curve(D_losses, G_losses, output_path):
    plt.figure()

    x = range(1, len(D_losses)+1)
    plt.plot(x, D_losses, color='orange', label='D loss')
    plt.plot(x, G_losses, color='blue', label='G loss')
    output_path = os.path.join(output_path, 'loss.png')
    plt.savefig(output_path)

#draw curve of loss
def save_curve(start_epoch, end_epoch, output_path):
    epoch = end_epoch-start_epoch+1
    D_losses = []
    D_fake_losses = []
    D_real_losses = []
    D_gp_losses = []
    G_losses = []
    txt = os.path.join(output_path, 'loss.txt')
    with open(txt, 'r') as f:
        for line in f.readlines():
            if len(line) < 7:
                continue
            D_fake_loss = line.split(' ')[0]
            D_real_loss = line.split(' ')[1]
            D_gp_loss = line.split(' ')[2]
            D_loss = line.split(' ')[3]
            G_loss = line.split(' ')[4]
            D_fake_losses.append(D_fake_loss)
            D_real_losses.append(D_real_loss)
            D_gp_losses.append(D_gp_loss)
            D_losses.append(D_loss)
            G_losses.append(G_loss)
    D_losses = np.array(D_losses).astype(np.float)
    D_fake_losses = np.array(D_fake_losses).astype(np.float)
    D_real_losses = np.array(D_real_losses).astype(np.float)
    D_gp_losses = np.array(D_gp_losses).astype(np.float)
    G_losses = np.array(G_losses).astype(np.float)
    total = len(D_losses)
    circle = total // epoch
    D_final = []
    D_fake_final = []
    D_real_final = []
    D_gp_final = []
    G_final = []
    for i in range(0, epoch):
        D_sum = 0
        D_fake_sum = 0
        D_real_sum = 0
        D_gp_sum = 0
        G_sum = 0
        for j in range(0, circle):
            D_sum += D_losses[i * circle + j]
            D_fake_sum += D_fake_losses[i * circle + j]
            D_real_sum += D_real_losses[i * circle + j]
            D_gp_sum += D_gp_losses[i * circle + j]
            G_sum += G_losses[i * circle + j]
        D_final.append(D_sum / circle)
        D_fake_final.append(D_fake_sum / circle)
        D_real_final.append(D_real_sum / circle)
        D_gp_final.append(D_gp_sum / circle)
        G_final.append(G_sum / circle)
    D_final = np.array(D_final).astype(np.float)
    D_fake_final = np.array(D_fake_final).astype(np.float)
    D_real_final = np.array(D_real_final).astype(np.float)
    D_gp_final = np.array(D_gp_final).astype(np.float)
    G_final = np.array(G_final).astype(np.float)
    x = range(start_epoch, start_epoch+epoch)
    plt.figure()
    plt.plot(x, D_final, color='red', label='D loss(' + str(circle) + ')')
    plt.plot(x, D_fake_final, color='orange', label='D fake loss(' + str(circle) + ')')
    plt.plot(x, D_real_final, color='yellow', label='D real loss(' + str(circle) + ')')
    plt.plot(x, D_gp_final, color='grey', label='D gp loss(' + str(circle) + ')')
    plt.plot(x, G_final, color='blue', label='G loss(' + str(circle) + ')')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_path, 'loss.png'))

def write_loss(D_loss, G_loss, D_fake, D_real, D_gp, output_path):
    with open(os.path.join(output_path, 'loss.txt'), 'a') as f:
        f.write(str(round(D_fake.item(),4)) + ' ' + str(round(D_real.item(),4)) + ' ' + str(round(D_gp.item(),4)) + ' ' + str(round(D_loss.item(),4)) + ' ' + str(round(G_loss.item(),4)) + '\n')
