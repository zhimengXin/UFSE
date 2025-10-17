import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# 定义编码器
class EncoderVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EncoderVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)
        #print("fc1 weight device:", self.fc1.weight.device)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim).to(device)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim).to(device)

    def forward(self, x):        
        h = torch.relu(self.fc1(x))
        #print()
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

# 定义解码器
class DecoderVAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(DecoderVAE, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(z))

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encodervae = EncoderVAE(input_dim, hidden_dim, latent_dim).to(device)
        self.decodervae = DecoderVAE(latent_dim, hidden_dim, input_dim).to(device)

    def forward(self, x):
        mu, logvar = self.encodervae(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_recon = self.decodervae(z)
        return x_recon, mu, logvar

# 定义损失函数
def vae_loss(x_recon, x, mu, logvar):
    x = torch.clamp(x, 0, 1)
    #recon_loss = F.binary_cross_entropy(x_recon, x, reduction ='sum')
    recon_loss = F.mse_loss(x_recon, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# # 设置训练参数
# input_dim = 784  # 28x28
# hidden_dim = 20
# latent_dim = 20
# lr = 1e-2
# batch_size = 512
# epochs = 10

# # 加载数据集
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# train_dataset = datasets.MNIST(root = './data', train = True, transform = transform, download = True)
# train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

# # 初始化模型和优化器
# model = VAE(input_dim, hidden_dim, latent_dim).to(device)
# optimizer = optim.Adam(model.parameters(), lr = lr)

# # 训练VAE模型
# model.train()
# for epoch in range(epochs):
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):
#         #data = data.view(-1, input_dim).to(device)
#         data = data.to(device)
#         optimizer.zero_grad()
#         new_data = data.view(512, -1)        
#         x_recon, mu, logvar = model(new_data)
#         loss = vae_loss(x_recon, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#     print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}') # type: ignore

# # 保存训练好的模型
# torch.save(model.state_dict(), 'vae.pth')

