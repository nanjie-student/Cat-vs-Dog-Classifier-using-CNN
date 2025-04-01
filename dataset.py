from torchvision import datasets, transforms # type: ignore
from torch.utils.data import DataLoader, Subset
import torch

def get_cat_dog_dataloader(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # cat = 3, dog = 5
    cat_dog_indices = [i for i, label in enumerate(cifar10.targets) if label in [3, 5]]
    cat_dog_subset = Subset(cifar10, cat_dog_indices)

    # 包装一下 subset，转 label: cat=0, dog=1
    class WrappedDataset(torch.utils.data.Dataset):
        def __init__(self, subset):
            self.subset = subset

        def __getitem__(self, index):
            x, y = self.subset[index]
            y = 0 if y == 3 else 1
            return x, y

        def __len__(self):
            return len(self.subset)

    dataset = WrappedDataset(cat_dog_subset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
