'''
This script implements the training procedure.
'''
import time
import os
import copy
from dataset import *
from utils import *
from draw import *


if __name__ == "__main__":

    # Detect if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parameters
    out_dir = './res'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('"{}" is created.'.format(out_dir))
    else:
        print('"{}" exists.'.format(out_dir))


    # Dataset
    """
    Test on VOC
    """
    # voc2012_train = VOC('/Users/erica/Dataset/Pascal/2012_train_short.txt', IMG_WIDTH, IMG_HEIGHT)
    voc2012_val = VOC('/Users/erica/Dataset/Pascal/2012_val_short.txt', IMG_WIDTH, IMG_HEIGHT)
    dataloader = DataLoader(voc2012_val, batch_size=1)

    # Model
    yolo_model = build_darknet()
    yolo_model = yolo_model.to(device)

    yolo_model.load_state_dict(torch.load('best_model_weight.pth'))  # weights
    print('Weights loaded.')

    yolo_model.eval()
    print('Evaluation mode.')

    since = time.time()
    y_out_epoch = torch.Tensor()  # record all output in a single epoch
    img_name_epoch = []

    running_loss = 0

    for i, (image_batch, label_batch, img_name_batch) in enumerate(dataloader):

        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        # forward pass
        y_out = yolo_model(image_batch)
        y_out_epoch = torch.cat((y_out_epoch, y_out), 0)
        img_name_epoch += img_name_batch

        # compute loss
        loss = calc_loss(y_out.clone(), label_batch.clone())
        running_loss += loss.item() * image_batch.size(0)

        print('Batch = {}\tLoss = {:.4f}'.format(i, loss.item()))

    # evaluation
    threshold = 0.5
    det = prediction2detection(y_out_epoch, img_name_epoch)
    ground_truth = ground_truth_detection(voc2012_val.label_list)
    res = evaluate_IOU(det, ground_truth)
    res_tp_fp = evaluate_TP_FP(res, threshold)
    results, acc_tps, acc_fps, precisions, recalls = evaluate_precision_recall(res_tp_fp, threshold, ground_truth)

    mAP = calc_mean_average_precision(precisions, recalls)
    epoch_loss = running_loss / len(dataloader.dataset)

    time_elapsed = time.time() - since

    print('\tLoss = {:.4f}\tmAP = {:.4f}\ttime_elapsed = {:.2f}\n'.format(epoch_loss, mAP, time_elapsed))
    print('Testing completed.')
    print('mAP = {:4f}'.format(mAP))

    # visualization
    visualize(y_out_epoch, img_name_epoch, voc2012_val.image_list, out_dir)
    print('Visualization completed.')
    print('done.')