'''
This script implement Model Parallel.

Output to folders:
                ./log
                ./plot
                ./checkpoints
                ./weights
'''

import time
import copy
import os.path
import matplotlib.pyplot as plt
from dataset import *
from model_parallel import *
from draw import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR


# def train(num_epoch, dataloader, model, optimizer, learning_rate=1e-4, scheduler=None,
#           phases=['train'], use_float64=True, checkpoint_interval=10):
def train(num_epoch, train_txt, val_txt, model, optimizer, learning_rate=1e-4, scheduler=None,
          phases=['train'], use_float64=True, use_visualization=False, checkpoint_interval=10):
    # suffix
    if scheduler is not None:
        suffix = '_lr={}_ep={}_w'.format(learning_rate, num_epoch)
    else:
        suffix = '_lr={}_ep={}_wo'.format(learning_rate, num_epoch)

    # Outputs
    fd_log = './log'
    if not os.path.exists(fd_log):
        os.mkdir(fd_log)
        print('{} not exists, it has been created.'.format(fd_log))

    log_file = 'log' + suffix + '.txt'
    if os.path.exists(log_file):
        os.remove(log_file)
        print('"{}" exists, it has been removed.'.format(log_file))

    fd_plot = './plot'
    if not os.path.exists(fd_plot):
        os.mkdir(fd_plot)
        print('{} not exists, it has been created.'.format(fd_plot))

    fd_checkpoints = './checkpoints'
    if not os.path.exists(fd_checkpoints):
        os.mkdir(fd_checkpoints)
        print('{} not exists, it has been created.'.format(fd_checkpoints))

    fd_weights = './weights'
    if not os.path.exists(fd_weights):
        os.mkdir(fd_weights)
        print('{} not exists, it has been created.'.format(fd_weights))


    # Dataset & Dataloader
    voc2012_train = VOC(train_txt, IMG_WIDTH, IMG_HEIGHT, data_transform)
    voc2012_val = VOC(val_txt, IMG_WIDTH, IMG_HEIGHT, data_transform)

    dataloader = dict()
    dataloader['train'] = DataLoader(voc2012_train, batch_size=batch_size, shuffle=True)
    dataloader['val'] = DataLoader(voc2012_val, batch_size=batch_size, shuffle=True)

    # Train
    epoch_loss_hist = dict()
    epoch_mAP_hist = dict()
    for elem in phases:
        epoch_loss_hist[elem] = []
        epoch_mAP_hist[elem] = []

    best_mAP = -1000.0
    min_loss = 1e10

    for epoch in range(num_epoch):
        since = time.time()
        print('\n-----------------------------------------')
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        y_out_epoch = torch.Tensor().to("cuda:1")  # record all output in a single epoch
        if use_float64:
            y_out_epoch = y_out_epoch.double()
        img_name_epoch = []

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0

            # Step 1: forward for all, and backward if in train phase
            for i, (image_batch, label_batch, img_name_batch) in enumerate(dataloader[phase]):

                image_batch = image_batch.to('cuda:0')
                label_batch = label_batch.to('cuda:1')

                if use_float64:
                    image_batch = image_batch.double()
                    label_batch = label_batch.double()

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # 1.1 forward
                    y_out = model(image_batch)
                    print('y_out mean = ', torch.mean(y_out))
                    print('y_out std = ', torch.std(y_out))

                    y_out_epoch = torch.cat((y_out_epoch, y_out), 0)
                    img_name_epoch += img_name_batch

                    # loss
                    loss = calc_loss(y_out.clone(), label_batch.clone(), device, use_float64)
                    running_loss += loss.item() * image_batch.size(0)

                    if scheduler is not None:
                        scheduler.step(loss)  # lr scheduler

                    # 1.2 backward
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)  # gradient clip
                        optimizer.step()

                print('{}\tEpoch = {}\tBatch = {}\tLoss = {:.4f}'.format(phase, epoch, i, loss.item()))

            # Step 2: evaluation
            iou_threshold = 0.5
            det = prediction2detection(y_out_epoch, img_name_epoch)

            if phase == 'train':
                ground_truth = ground_truth_detection(voc2012_train.label_list)
            else:
                ground_truth = ground_truth_detection(voc2012_val.label_list)

            res = evaluate_IOU(det, ground_truth, device, use_float64=use_float64)
            res_tp_fp = evaluate_TP_FP(res, iou_threshold)

            results, acc_tps, acc_fps, precisions, recalls = evaluate_precision_recall(res_tp_fp, iou_threshold, ground_truth)
            mAP = calc_mean_average_precision(precisions, recalls)

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_loss_hist[phase].append(epoch_loss)
            epoch_mAP_hist[phase].append(mAP)

            time_elapsed = time.time() - since

            # Step 3: log, save weights
            # log
            print('{}\tLoss = {:.4f}\tmAP = {:.4f}\ttime_elapsed = {:.2f} s\n'.format(phase, epoch_loss, mAP, time_elapsed))
            f = open(os.path.join(fd_log, log_file), 'a')
            f.write('{}\tEpoch = {}\tLoss = {:.4f}\tmAP = {:.4f}\ttime_elapsed = {:.2f} s\n'.format(
                phase, epoch, epoch_loss, mAP, time_elapsed))
            f.close()

            # save weights
            # - min loss
            if phase == 'val' and epoch_loss < min_loss:
                torch.save(copy.deepcopy(model.state_dict()), os.path.join(fd_weights, 'min_loss_weights' + suffix + '.pth'))
                torch.save(copy.deepcopy(model.state_dict()), os.path.join(fd_weights, 'min_loss_weights.pth'))
                min_loss = epoch_loss
                print('[val] A smaller loss is found. \nModel saved.')

            # - best mAP
            # if mAP > best_mAP:
            if phase == 'val' and mAP > best_mAP:
                best_mAP = mAP
                torch.save(copy.deepcopy(model.state_dict()),
                           os.path.join(fd_weights, 'best_model_weights' + suffix + '.pth'))
                print('[val] Best model is saved.')

            # - checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                torch.save(copy.deepcopy(model.state_dict()),
                           os.path.join(fd_checkpoints, 'checkpoint_weights_ep{}.pth'.format(epoch+1)))
                print('Checkpoint is saved at {}.'.format(fd_checkpoints))

            # Step 4: visualization
            if use_visualization:
                visualize(y_out_epoch, img_name_epoch, voc2012_train.image_list, fd_plot)
                print('Visualization completed.')
            
        # plot history
        plt.figure(1)
        for p in phases:
            color = 'r' if p == 'train' else 'm'
            plt.plot(range(epoch+1), epoch_loss_hist[p], color)
        plt.title('Loss         lr = {}'.format(learning_rate))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(phases)
        plt.savefig(os.path.join(fd_plot, 'loss_history' + suffix + '.png'))

        plt.figure(2)
        for p in phases:
            color = 'b' if p == 'train' else 'g'
            plt.plot(range(epoch+1), epoch_mAP_hist[p], color)
        plt.title('mAP         lr = {}'.format(learning_rate))
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend(phases)
        plt.savefig(os.path.join(fd_plot, 'mAP_history' + suffix + '.png'))    
    return min_loss, best_mAP


if __name__ == "__main__":
    # Memory / Storage
    os.system('rm -rf checkpoints/ log/ weights/ plot/')
    torch.cuda.empty_cache()

    # Device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Parameter
    num_epoch = 50
    batch_size = 32
    use_float64 = False
    use_scheduler = False
    use_bn = True
    learning_rate = 5e-6

    # Weights
    model_weights = './results/1e-5_ep=1-10/checkpoint_weights_ep10.pth'
    # model_weights = None

    # Dataset
    phases = ['train', 'val']
    train_txt = '/home/bizon/Dataset/VOC_yolo_format/2007_train.txt'
    val_txt = '/home/bizon/Dataset/VOC_yolo_format/2007_val.txt'
    # train_txt = '/home/bizon/Dataset/VOC_yolo_format/2012_sanity.txt'
    # val_txt = '/home/bizon/Dataset/VOC_yolo_format/2012_sanity.txt'

    # Model
    yolo_model = build_darknet_parallel(path=model_weights, use_bn=use_bn)
    if use_float64:
        yolo_model = yolo_model.double()
    # Optimizer
    optimizer = torch.optim.Adam(yolo_model.parameters(), lr=learning_rate, eps=1e-6)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True) \
    #     if use_scheduler else None

    scheduler = StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=-1) if use_scheduler else None

    ################### Train ###################
    print('\n\nlearning rate = ', learning_rate)

    min_loss, best_mAP = train(num_epoch=num_epoch, train_txt=train_txt, val_txt=val_txt, phases=phases, model=yolo_model,
                               optimizer=optimizer, learning_rate=learning_rate, scheduler=scheduler,
                               checkpoint_interval=1, use_float64=use_float64)

    print('=======================================')
    print('Training completed.')
    print('Best mAP = {:4f}'.format(best_mAP))
    print('Min loss = {:4f}'.format(min_loss))
    print('Done.')




