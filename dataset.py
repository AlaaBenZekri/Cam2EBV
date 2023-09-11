import utils
import os
from torchvision.io import read_image
from torch import tensor
from torch.utils.data import Dataset

class Cam2BEVDataset(Dataset):
    def __init__(self, input_dirs, label_dir, input_shape, output_shape, input_palette, output_palette, device):
        self.input_dirs = input_dirs
        self.label_dir = label_dir
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_palette = input_palette
        self.output_palette = output_palette
        self.device = device
        self.images = os.listdir(input_dirs[0])
		
    def __len__(self):
        return len(self.images)
		
    def __getitem__(self, idx):
        inputs = []
        for i in self.input_dirs:
            image_path = os.path.join(i, self.images[idx])
            image = utils.load_image(image_path)
            image = utils.resize_image(image, self.input_shape)
            image = utils.one_hot_encode_image(image, self.input_palette)
            image = tensor(image).to(self.device)
            inputs.append(image)
        label_path = os.path.join(self.label_dir, self.images[idx])
        label = utils.load_image(label_path)
        label = utils.resize_image(label, self.output_shape)
        label = utils.one_hot_encode_image(label, self.output_palette)
        label = tensor(label).to(self.device)
        return (self.images[idx], inputs, label)
