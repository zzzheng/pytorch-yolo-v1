'''
This script implements the training procedure.
'''
import time
import copy
import os.path
import matplotlib.pyplot as plt
from dataset import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR


# Memory
torch.cuda.empty_cache()

# Detect if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('run in {}'.format(device))



if __name__ == "__main__":

    # Parameters
    num_epoch = 100
    learning_rate = 1e-2
    batch_size = 2

    # model_weights = 'min_loss_weights.pth'
    model_weights = None

    use_scheduler = True



    # Setting
    # phases = ['train', 'val']
    phases = ['train']

    # train_txt = '/home/bizon/Dataset/VOC_yolo_format/2012_train.txt'
    # val_txt = '/home/bizon/Dataset/VOC_yolo_format/2012_val.txt'

    # train_txt = '/home/bizon/Dataset/VOC_yolo_format/2012_train_short.txt'
    # val_txt = '/home/bizon/Dataset/VOC_yolo_format/2012_train_short.txt'

    train_txt = '/home/bizon/Dataset/VOC_yolo_format/2012_sanity.txt'
    val_txt = '/home/bizon/Dataset/VOC_yolo_format/2012_sanity.txt'

    # Memory
    torch.cuda.empty_cache()

    # Dataset
    voc2012_train = VOC(train_txt, IMG_WIDTH, IMG_HEIGHT, data_transform)
    voc2012_val = VOC(val_txt, IMG_WIDTH, IMG_HEIGHT, data_transform)

    dataloader = dict()
    dataloader['train'] = DataLoader(voc2012_train, batch_size=batch_size)
    dataloader['val'] = DataLoader(voc2012_val, batch_size=batch_size)

    # Model
    yolo_model = build_darknet(path=model_weights)
    yolo_model = nn.DataParallel(yolo_model, device_ids=[1])

    # Optimizer
    optimizer = torch.optim.Adam(yolo_model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(yolo_model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


    # Log
    log_file = 'log.txt'
    if os.path.exists(log_file):
        os.remove(log_file)
        print('"{}" exists, it has been removed.'.format(log_file))

    epoch_loss_hist = dict()
    for elem in phases:
        epoch_loss_hist[elem] = []

    # Train
    best_mAP = 0.0
    min_loss = 1e10

    for epoch in range(num_epoch):
        since = time.time()
        print('\n-----------------------------------------')
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        y_out_epoch = torch.Tensor().to(device)  # record all output in a single epoch
        img_name_epoch = []



        for phase in phases:
            if phase == 'train':
                yolo_model.train()
            else:
                yolo_model.eval()

            running_loss = 0

            # Step 1: forward / backward
            for i, (image_batch, label_batch, img_name_batch) in enumerate(dataloader[phase]):

                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # foward pass
                    y_out = yolo_model(image_batch)
                    y_out_epoch = torch.cat((y_out_epoch, y_out), 0)
                    y_out_epoch = y_out_epoch.to(device)  # ZZ added
                    img_name_epoch += img_name_batch

                    # compute loss
                    loss = calc_loss(y_out.clone(), label_batch.clone())
                    running_loss += loss.item() * image_batch.size(0)

                    if use_scheduler:
                        scheduler.step(loss)  # lr scheduler

                    # backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                print('{}\tEpoch = {}\tBatch = {}\tLoss = {:.4f}'.format(phase, epoch, i, loss.item()))



            # Step 2: evaluation
            threshold = 0.5
            det = prediction2detection(y_out_epoch, img_name_epoch)

            if phase == 'train':
                ground_truth = ground_truth_detection(voc2012_train.label_list)
            else:
                ground_truth = ground_truth_detection(voc2012_val.label_list)

            res = evaluate_IOU(det, ground_truth)
            res_tp_fp = evaluate_TP_FP(res, threshold)
            results, acc_tps, acc_fps, precisions, recalls = evaluate_precision_recall(res_tp_fp, threshold, ground_truth)

            mAP = calc_mean_average_precision(precisions, recalls)
            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_loss_hist[phase].append(epoch_loss)

            time_elapsed = time.time() - since

            # Step 3: log, save weights
            # log
            print('{}\tLoss = {:.4f}\tmAP = {:.4f}\ttime_elapsed = {:.2f} s\n'.format(phase, epoch_loss, mAP, time_elapsed))
            f = open(log_file, 'a')
            f.write('{}\tEpoch = {}\tLoss = {:.4f}\tmAP = {:.4f}\ttime_elapsed = {:.2f} s\n'.format(phase, epoch, epoch_loss, mAP, time_elapsed))
            f.close()

            # weights
            # - min loss
            if epoch_loss < min_loss:
                torch.save(copy.deepcopy(yolo_model.state_dict()), 'min_loss_weights.pth')
                min_loss = epoch_loss
                print('a smaller loss is found.')
            # - best mAP
            if phase == 'val' and mAP > best_mAP:  
                best_mAP = mAP
                torch.save(copy.deepcopy(yolo_model.state_dict()), 'best_model_weights.pth')

            # plot history
            for p in phases:
                plt.plot(range(epoch+1), epoch_loss_hist[p])
            plt.title('lr = {}'.format(learning_rate))
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(phases)
            plt.savefig('loss_history.png')

    print('Training completed.')
    print('Best mAP = {:4f}'.format(best_mAP))
    print('Done.')
