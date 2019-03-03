from torch.utils.data import Dataset
import torch

class SegDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index].type(torch.LongTensor)

        return x, y

    def __len__(self):
        return len(self.data)

class BBoxDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # file_name = self.data[index]['filepath']
        # width = self.data[index]['width']
        # height = self.data[index]['height']
        # x1 = self.data[index]['bboxes']['x1']
        # x2 = self.data[index]['bboxes']['x2']
        # y1 = self.data[index]['bboxes']['y1']
        # y2 = self.data[index]['bboxes']['y2']

        return self.data[index]

    def __len__(self):
        return len(self.data)