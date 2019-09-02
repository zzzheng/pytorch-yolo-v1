from torch.utils.data import Dataset, DataLoader
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
from train import *
from torchvision import transforms
from torchvision.transforms import Normalize


data_transform = transforms.Compose([
    # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # for imageNet
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # good
])

class VOC(Dataset):
    """
        Pascal VOC dataset.
    Note:
        using YOLO label format
        https://github.com/pjreddie/darknet
    Example:
        voc2012 = VOC('2012_train_short.txt', 448, 448)
        dataloader = DataLoader(voc2012, batch_size=4)
        I = voc2012[0][0]
        I = I.permute(1, 2, 0)
        plt.imshow(I)
        plt.show()
    """

    def __init__(self, txt_file, img_width=None, img_height=None, transform=None):
        """

        :param txt_file: all image directories
        """
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        self.image_list = [i.rstrip('\n') for i in lines]
        self.label_list = [str.replace('JPEGImages', 'labels').replace('.jpg', '.txt')
                           for str in self.image_list]

        self.img_width = img_width
        self.img_height = img_height
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # get image
        image = io.imread(self.image_list[idx])

        if self.img_width and self.img_height:
            image = resize(image, (self.img_width, self.img_height))
            image = torch.Tensor(image).permute(2, 0, 1)  # pytorch format: C W H

        if self.transform:
            image = self.transform(image)

        # get label
        label = read_labels(self.label_list[idx])
        # convert to S*S*5 Tensor with format <x> <y> <w> <h> <cls>
        label = labels2tensor(label)

        # get filename
        filename = self.image_list[idx].split('/')[-1]

        return image, label, filename


if __name__ == "__main__":



    """
    # Train on VOC
    """
    voc2012 = VOC('/home/bizon/Dataset/VOC_yolo_format/2012_train_short.txt', IMG_WIDTH, IMG_HEIGHT, data_transform)
    dataloader = DataLoader(voc2012, batch_size=4)

    # Model
    yolo_model = build_darknet()
    yolo_model.train()

    # Optimize
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(yolo_model.parameters(), lr=learning_rate)

    num_epoch = 1
    y_out_epoch = torch.Tensor()  # record all output in a single epoch
    img_name_epoch = []
    for epoch in range(num_epoch):
        for i, (image_batch, label_batch, img_name_batch) in enumerate(dataloader):

            print('batch = ', i)
            print('image  = ', image_batch.size())
            print('label =', label_batch.size())

            # foward pass
            y_out = yolo_model(image_batch)
            y_out_epoch = torch.cat((y_out_epoch, y_out), 0)
            img_name_epoch += img_name_batch
            # compute loss
            loss = calc_loss(y_out.clone(), label_batch.clone())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('\nEpoch = ', epoch, 'Batch = ', i, 'Loss = ', loss.item())

        # evaluation
        det = prediction2detection(y_out_epoch, img_name_epoch)
        ground_truth = ground_truth_detection(voc2012.label_list)
        res = evaluate_IOU(det, ground_truth)
        res_tp_fp = evaluate_TP_FP(res, 0.5)
        results, acc_tps, acc_fps = evaluate_precision_recall(res_tp_fp, 0.5, ground_truth)

        print('Epoch {} done.'.format(epoch))
        print('Acc TP for all classes = {} \n, Acc FP for all classes = {}\n'.format(acc_tps, acc_fps))
    print('Done.')






