import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader

from network import generator, discriminator
from dataset import ImageDataset
from train import train

latent_dim = 100 #input vector of G
classes_dim = 14 #class num
img_size = 128 #generator imge size
lr = 0.0002 #lr
batch_size = 64 #batch size
beta1 = 0.5 #Adam beta1
beta2 = 0.999 #Adam beta2
lambda_adv = classes_dim #lambda of valid loss
lambda_gp = 0.5 #lambda of gp loss
start_epochs = 0 #train epochs
end_epochs = 100
adversarial_loss = torch.nn.BCELoss() #valid loss function
auxiliary_loss = torch.nn.BCELoss() #label loss function
train_mode = 'train_all' #data folder
root = './data' #train data root path
output_path = './output29' #output path
is_load = False
#cuda
if torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')

G = generator(latent_dim, classes_dim)
D = discriminator(3, classes_dim)
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

G.to(device)
D.to(device)
'''
output17: 0.0003--(0.8/epoch2)-->epoch126 0.0003--(0.8/epoch10)-->epoch180 0.00025--(0.8/epoch15)-->epoch226
output18: 0.0002-->iteration50000 0.0002--(0.9/epoch1)
output19: 0.0001--(0.95/epoch1)-->epoch100
output20: network last 2 bias (delete false) 0.0002-->iteration50000 0.0002--(0.1/epoch5) 
output21: network some relu (add True) 0.0002--(0.8/epoch10)
output22: lambda_adv=14 0.0002--(0.1/epoch10)
output23: 0.0002-->iteration50000 0.0002--(0.1/epoch1) 
output24: add 0.2,0.8,True... 0.0002-->iteration50000 0.0002--(0.1/epoch1)
output25: no_gen 0.0002-->iteration50000 0.0002--(0.1/epoch1)
output26: delete 0.8 0.0002-->iteration50000 0.0002--(0.1/epoch1)
output27: add bias=False 0.0002-->iteration50000 0.0002--(0.1/epoch1)
output28: add instanceNorm2d affine=True 0.0002-->iteration50000 0.0002--(0.1/epoch1)
output29: delete instanceNorm2d, 0.2, add 0.8 0.0002-->iteration50000 0.0002--(0.1/epoch1)
'''

if is_load:
    G_load_path = os.path.join('./output18', 'G(1gp_2gen_bn)_epoch44.pth')
    D_load_path = os.path.join('./output18', 'D(1gp_2gen_bn)_epoch44.pth')
    G_state = torch.load(G_load_path)
    D_state = torch.load(D_load_path)
    G.load_state_dict(G_state)
    D.load_state_dict(D_state)

#print(G)
#print(D)

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, ), std=(0.5, ))
])
train_dataloader = DataLoader(
    ImageDataset(root=root, transforms_=transform, mode=train_mode, start_year=12),
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    drop_last=True
)
train(G=G, D=D,
      train_dataset=train_dataloader,
      start_epochs=start_epochs, end_epochs=end_epochs, batch_size=batch_size,
      optimizer_G=optimizer_G, optimizer_D=optimizer_D,
      label_criterion=auxiliary_loss, valid_criterion=adversarial_loss,
      latent_dim=latent_dim, classes_dim=classes_dim,
      output_path=output_path,
      device=device
      )
