'''dataset loader'''
import torch.utils.data
from data.base_dataset import BaseDataset
from data.arto_dataset import ARTODataset
class CustomDataset(object):
    """User-defined dataset
    
    Example usage:
        >>> from data import CustomDataset
        >>> dataset = CustomDataset(opt, is_for_train)
        >>> dataloader = dataset.load_data()
    """
    def __init__(self, opt, is_for_train):
        self.opt = opt
        if opt.dataset_mode.lower() == 'arto':
            self.dataset = ARTODataset(opt,is_for_train)
            print("dataset [%s] was created" % type(self.dataset).__name__)
        else:
            raise ValueError(opt.dataset_mode, "not implmented.")
        
        print(len(self.dataset))
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=is_for_train,
            num_workers=int(opt.num_threads),
            drop_last=False)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)