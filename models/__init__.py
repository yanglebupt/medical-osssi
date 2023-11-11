from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,], [0.5,])
                   ])

class MyDataset(Dataset):
    def __init__(self, features, labels, maxvalues):
        data = ((features / maxvalues) - 0.5) / 0.5
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.float32)
    
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        return self.data[idx, :], self.labels[idx]