# Libraries
import numpy as np
import torch
import torchvision as tv
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# Configs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 500
lr = 2e-4
classes = 10
channels = 1
img_size = 28
latent_dim = 100
log_interval = 1

# Load Dataset
train = tv.datasets.FashionMNIST(
    root="../masalan2/data",
    train=True,
    download=False,
    transform=ToTensor()
)
trainLoader = DataLoader(train,batch_size=batch_size,shuffle=True)


# Generator
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.img_shape = (channels,img_size,img_size)
        self.label_embedding = nn.Embedding(classes,classes)
        self.model = nn.Sequential(
            *self._layer(latent_dim+classes,128,False),
            *self._layer(128,256),
            *self._layer(256,512),
            *self._layer(512,1024),
            nn.Linear(1024,int(np.prod(self.img_shape))), # 1*28*28
            nn.Tanh()
        )
    def _layer(self,size_in,size_out,normalize=True):
        layers = [nn.Linear(size_in,size_out)]
        if normalize:
            layers.append(nn.BatchNorm1d(size_out))
        layers.append(nn.LeakyReLU(0.2,inplace=True))
        return layers
    def forward(self,noise,labels):
        z = torch.cat((self.label_embedding(labels),noise),-1)
        x = self.model(z)
        x.view(x.size(0),*self.img_shape)
        return x
    
# Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.img_shape = (channels,img_size,img_size)
        self.label_embedding = nn.Embedding(classes,classes)
        self.adv_loss = nn.BCELoss()
        self.model = nn.Sequential(
            *self._layer(classes+int(np.prod(self.img_shape)),1024,False,True),
            *self._layer(1024,512,True,True),
            *self._layer(512,256,True,True),
            *self._layer(256,128,False,False),
            *self._layer(128,1,False,False),
            nn.Sigmoid()
        )
    def _layer(self,size_in,size_out,drop_out=True,act_func=True):
        layers = [nn.Linear(size_in,size_out)]
        if drop_out:
            layers.append(nn.Dropout(0.4))
        if act_func:
            layers.append(nn.LeakyReLU(0.2,inplace=True))
        return layers
    def forward(self,image,labels):
        x = torch.cat((image.view(image.size(0), -1), self.label_embedding(labels)), -1)
        x = self.model(x)
        return x
    def loss(self,output,label):
        return self.adv_loss(output,label)


gen = generator() # define generator
disc = discriminator() # define discriminator

optG = torch.optim.Adam(gen.parameters(),lr=lr,betas=(0.5,0.999))
optD = torch.optim.Adam(disc.parameters(),lr=lr,betas=(0.5,0.999))

if __name__ == "__mane__":
    # Train
    gen.train() ; disc.train()

    for epoch in range(epochs):
        for i , (data,target) in enumerate(trainLoader):
            data , target = data.to(device) , target.to(device)
            real_label = torch.full((batch_size,1),1.,device=device)
            fake_label = torch.full((batch_size,1),0.,device=device)

            
            # Train Generator
            optG.zero_grad() 
            noise = torch.randn(batch_size,latent_dim,device=device)
            x_fake_labels = torch.randint(0,classes,(batch_size,),device=device) # generate a random class label
            x_fake = gen(noise , x_fake_labels) # generate a fake image
            y_fake_gen = disc(x_fake , x_fake_labels) # predict labels of generated images
            g_loss = disc.loss(y_fake_gen , real_label) 
            g_loss.backward()
            optG.step()
            # Train discriminator
            try:    
                optD.zero_grad()
                y_real = disc(data,target) # train discriminator on real data
                d_real_loss = disc.loss(y_real,real_label)
                y_fake_d = disc(x_fake.detach() , x_fake_labels) # predict labels of generated images for train discriminator
                d_fake_loss = disc.loss(y_fake_d , fake_label)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optD.step()
            except: continue


        print('Epoch [{}/{}] loss_D: {:.4f} loss_G: {:.4f}'.format(
                        epoch+1, epochs,
                        d_loss.mean().item(),
                        g_loss.mean().item()))


    checkpoint = { 
        'epoch': epochs,
        'generator': gen.state_dict(),
        'discriminator': disc.state_dict(), 
        'optimizer_generator': optG.state_dict(),
        'optimizer_discriminator':optD.state_dict(),
        'lr_sched': lr}
    torch.save(checkpoint, '../models/GAN.pth')