import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


# by default the model has .58 M parameters you can adjust settings based what hardware you have 


# Util  parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_weights_path = 'model_cifar10.pt'
train = True # !!!IMPORTANT: set to TRUE if you want the model to train this will override you last weights in the same directory if Train is set to False it will load the weights you last generated!!!
show_example = True # this will show a single batch of example outputs


# Model parameters
patch_size = 2
embed_dim  = 96
depth = 5
n_heads = 12
mlp_ratio = 4.0
qvv_bias = True


# Training parameters
batch_size = 32
learning_rate = 3e-4
dropout =  0.2
attn_dropout = 0.2
step_size=10
gamma= 0.1
n_epochs = 10
data_augmentation = False


# Data Augmentation/ Normalization
if data_augmentation == False:
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
else:
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])




# Load the training and test sets
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)




trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)





# Patch Embedding Class
class Patchembed(nn.Module):
    def __init__(self, img_size, patch_size, in_chan, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chan, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.view(x.shape[0], self.proj.out_channels, -1)
        x = x.permute(0,2,1)
        return x


# Multiheaded Attention Class:
class Attention(nn.Module):
    def __init__(self,dim,n_heads=8,qkc_bias=True,attn_p=0.,proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim  = dim // n_heads
        self.scale = self.head_dim**-.5
        
        self.qvk = nn.Linear(dim,3*dim,bias=qkc_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self,x):
        n_samples, n_tokens, dim = x.shape

        qkv = self.qvk(x)
        qkv = qkv.reshape(n_samples,n_tokens,3,self.n_heads,self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]

        out = q@k.transpose(-2,-1)*self.scale
        out = self.attn_drop(out)
        out = F.softmax(out,dim=-1)@v
        out = out.transpose(1,2)
        out = out.reshape(n_samples,n_tokens,-1)
        x = self.proj(out)
        x = self.proj_drop(x)

        return(x)
    
# Multilayer Perceptron Class
class MLP(nn.Module):
    def  __init__(self,in_features,hidden_features,out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act= nn.GELU()
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(p)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



# Encoder Block
class Block(nn.Module):

    def __init__(self, dim, n_heads, mlp_ratio=4.0,qkv_bias=True,p=0.,attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim,n_heads,qkc_bias=qkv_bias,attn_p=attn_p,proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim*mlp_ratio)
        self.mlp = MLP(dim,hidden_features,dim)

    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


# Vision Transformer Class
class Visiontransformer(nn.Module):
    def __init__(self,img_size=32,patch_size=patch_size,in_chans=3,n_classes=10,embed_dim=embed_dim,depth=depth,n_heads=n_heads,mlp_ratio=mlp_ratio,qvv_bias=qvv_bias,p=dropout,attn_p=attn_dropout):
        super().__init__()
        self.n_patches = int((img_size // patch_size) ** 2)
        self.patch_embed = Patchembed(img_size=img_size,patch_size=patch_size,in_chan=in_chans,embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,1+self.n_patches,embed_dim))
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList([Block(dim=embed_dim,n_heads=n_heads,mlp_ratio=mlp_ratio,qkv_bias=qvv_bias,p=p,attn_p=attn_p) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim,n_classes)
        
    def forward(self,x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token,x),dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_final = x[:,0]
        x = self.head(cls_token_final)

        return x



model  = Visiontransformer().to(device)


# Prints number of parameters in Vision Transformer
print(sum(p.numel()for p in model.parameters())/1e6, 'M parameters')




criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)











if train == False:
    assert os.path.isfile(model_weights_path), "Weights file does not exist!"
    model.load_state_dict(torch.load(model_weights_path))

else:
    
    ### Training  Loop ###
    best_val_loss = float('inf')
    model.train()
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        
        
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            
        
            logits = model(data)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()*data.size(0)
            
    
        running_loss /= len(trainloader.dataset)
        scheduler.step()

        running_val_loss = 0.0
        correct = 0
        total = 0
        
        
        model.eval()
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                
                
                outputs = model(data)
                loss = criterion(outputs, target)
                
                running_val_loss += loss.item() * data.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        running_val_loss /= len(testloader.dataset)
        accuracy = 100 * correct / total
        if running_val_loss < best_val_loss:
            print(f'Validation loss decreased ({best_val_loss:.6f} --> {running_val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), 'model_cifar10.pt')
            best_val_loss = running_val_loss



        print(f'Epoch: {epoch+1} \tTraining Loss: {running_loss:.6f} \tValidation Loss: {running_val_loss:.6f}')
        print('Accuracy of the network on the test images: %d %%' % accuracy)









# Shows grid of images with labels if Show_Example is set to True
if show_example == True:

    model.eval()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    def imshow(img):
        img = img.cpu() / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    dataiter = iter(testloader)
    images, labels = next(dataiter)

    images = images.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)



    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(2, batch_size//2, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title("{} ({})".format(classes[predicted[idx]], classes[labels[idx]]),
                    color=("green" if predicted[idx]==labels[idx].item() else "red"))
    plt.show()
