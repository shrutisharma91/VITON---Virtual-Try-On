import os
from torch.utils.data import Dataset
from utils import load_image_as_numpy

class TryOnDataset(Dataset):
    """
    Minimal dataset enumerator.
    Expects:
      datasets/person/   -> person images
      datasets/clothes/  -> cloth images
    Pairs are matched by index order (sorted filenames).
    """

    def __init__(self, person_dir='datasets/person', cloth_dir='datasets/clothes', size=(256,256)):
        self.person_dir = person_dir
        self.cloth_dir = cloth_dir
        self.person_files = sorted([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        self.cloth_files = sorted([f for f in os.listdir(cloth_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        self.size = size
        self.length = min(len(self.person_files), len(self.cloth_files))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        person_path = os.path.join(self.person_dir, self.person_files[idx])
        cloth_path = os.path.join(self.cloth_dir, self.cloth_files[idx])
        person_np = load_image_as_numpy(person_path, size=self.size)  # H,W,3 uint8
        cloth_np = load_image_as_numpy(cloth_path, size=self.size)
        sample = {
            'person_np': person_np,
            'cloth_np': cloth_np,
            'person_name': self.person_files[idx],
            'cloth_name': self.cloth_files[idx]
        }
        return sample
