import torch
import time
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, RandomSampler


class AQPDataset(Dataset):
    def __init__(self, data_path ):
        super(AQPDataset, self).__init__()
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.data = pkl.load(f)
        
   
        self.metedata = self.data['metedata']
        self.AQdata = self.data['AQdata']
        self.AQStation_imformation = self.data['AQStation_imformation_before']
        self.meteStation_imformation = self.data['meteStation_imformation']


    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images[name]
        image = torch.Tensor(255-image) / 255
        image = image.unsqueeze(0)
        labels.append('eos')
        words = self.words.encode(labels)
        words = torch.LongTensor(words)
        return image, words


def get_AQ_dataset(params, use_aug):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    print(f"训练数据路径 images: {params['train_image_path']} labels: {params['train_label_path']}")
    print(f"验证数据路径 images: {params['eval_image_path']} labels: {params['eval_label_path']}")

    train_dataset = AQPDataset(params['train_data_path'])
    # eval_dataset = HMERDataset(params, params['eval_image_path'], params['eval_label_path'], words, is_train=False)

    train_sampler = RandomSampler(train_dataset)
    # eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    # eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
    #                           num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} ')
    
    return train_loader


def collate_fn(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, image_masks, labels, labels_masks


class Words:
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label


collate_fn_dict = {
    'collate_fn': collate_fn
}