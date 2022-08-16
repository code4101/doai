import torch
from torchvision import datasets, transforms

batch_size = 200

dst_dir = r'/home/chenkunze/data'

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(dst_dir, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(dst_dir, train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

from cv2 import imwrite

for x, y in train_loader:
    x.shape  # [batch, channel, height, width]
    y.shape  # [batch]
    imwrite('aaa.jpg', x[0, 0].numpy() * 255)  # ToTensor的时候变成[0, 1]，要放大回255
    break
