'''
This script implements the training procedure.
'''
from dataset import *


# Detect if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    # Parameters
    # optimizer
    num_epoch = 3
    learning_rate = 1e-4

    # Dataset
    """
    Train on VOC
    """
    voc2012_train = VOC('/Users/erica/Dataset/Pascal/2012_train_short.txt', IMG_WIDTH, IMG_HEIGHT)
    voc2012_val = VOC('/Users/erica/Dataset/Pascal/2012_val_short.txt', IMG_WIDTH, IMG_HEIGHT)

    dataloader = dict()
    dataloader['train'] = DataLoader(voc2012_train, batch_size=4)
    dataloader['val'] = DataLoader(voc2012_val, batch_size=4)

    # Model
    yolo_model = build_darknet()
    yolo_model = yolo_model.to(device)
    yolo_model.train()

    # Optimize
    optimizer = torch.optim.Adam(yolo_model.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        print('\n-----------------------------------------')
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        y_out_epoch = torch.Tensor()  # record all output in a single epoch
        img_name_epoch = []

        for phase in ['train', 'val']:
            if phase == 'train':
                yolo_model.train()
            else:
                yolo_model.eval()

            running_loss = 0

            for i, (image_batch, label_batch, img_name_batch) in enumerate(dataloader[phase]):

                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # foward pass
                    y_out = yolo_model(image_batch)
                    y_out_epoch = torch.cat((y_out_epoch, y_out), 0)
                    img_name_epoch += img_name_batch

                    # compute loss
                    loss = calc_loss(y_out.clone(), label_batch.clone())
                    running_loss += loss.item() * image_batch.size(0)

                    # backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                print('Epoch = ', epoch, '\tBatch = ', i, '\tLoss = ', loss.item())

            # evaluation
            threshold = 0.5
            det = prediction2detection(y_out_epoch, img_name_epoch)
            ground_truth = ground_truth_detection(voc2012_train.label_list)
            res = evaluate_IOU(det, ground_truth)
            res_tp_fp = evaluate_TP_FP(res, threshold)
            results, acc_tps, acc_fps, precisions, recalls = evaluate_precision_recall(res_tp_fp, threshold, ground_truth)
            mAP = calc_mean_average_precision(precisions, recalls)

            epoch_loss = running_loss / len(dataloader[phase].dataset)

            print('{}\tLoss = {}\tmAP = {}\n'.format(phase, epoch_loss, mAP))

    print('Done.')
