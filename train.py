'''
This script implements the training procedure.
'''
from model import *
from utils import *
# import math
# import sys
import torch
from dataset import *

LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5


def predict_one_bbox(P, G, device=torch.device("cpu"), use_float64=False):
    """
    Tensor version of predict_one_box
    Select ONE bounding box per grid cell.
    Note:
        YOLO predicts MULTIPLE bounding boxes per grid cell.
        At training time we only want one bounding box predictor to be responsible for each object.
        We assign one predictor to be “responsible” for predicting an object
        based on which prediction has the highest current IOU with the ground truth.

    :param P: Prediction tensor S*S*(B*5+C) with MULTIPLE bounding boxes per grid cell.
                format < <x> <y> <w> <h> <confidence> > * B + <cls_prob>

    :param G: GroundTruth tensor S*S*5
                format <x> <y> <w> <h> <class name>

    :return: Q: Prediction tensor S*S*(5+C) with SINGLE bounding box per grid cell.
                format  <x> <y> <w> <h> <confidence> <cls_prob>
    """
    if P.size(0) != S or P.size(1) != S or P.size(2) != 5*B+C:
        raise Exception("Tensor size not match")

    # convert to image coordinate [0,1]
    # #### Do ONLY once !!!
    P = convert_coord_cell2img(P)  # todo: not compatible
    Q = torch.zeros(S, S, 5+C)     # init
    if use_float64:
        Q = Q.double()

    select = torch.tensor(0).to(device)  # init

    for i in range(S):              # loop over each grid cell
        for j in range(S):

            # localization loss
            # boxes = torch.tensor([], dtype=torch.float32)          # store all boxes' position (x, y, w, h)
            # get all bbox assigned for this grid cell
            # format < <x> <y> <w> <h> <confidence>
            for b in range(B):
                if b == 0:
                    boxes = P[i, j, b*5: b*5+5].to(device)
                else:
                    # boxes.append((P[i, j, b*5], P[i, j, b*5+1], P[i, j, b*5+2], P[i, j, b*5+3], P[i, j, b*5+4]))
                    boxes = torch.stack((boxes, P[i, j, b*5: b*5+5])).to(device)

            # case 1: ground truth has bbox at this grid cell
            #         Select one box has the highest IOU with ground truth
            if len(G[i, j, :].nonzero()) > 1:
                # max_iou = 0  # init
                max_iou = torch.tensor([0.]).to(device)  # init
                if use_float64:
                    max_iou = max_iou.double()

                groundtruth_box = torch.clone(G[i, j, :4])

                for b in range(B):
                    # iou = calc_IOU(groundtruth_box, boxes[b][:-1])
                    iou = calc_IOU(groundtruth_box, boxes[b][:-1], device, use_float64) # use Tensor version

                    if iou > max_iou:
                        max_iou = iou
                        select = torch.tensor(b).to(device)

            # case 2: ground truth has NO bbox at this grid cell
            #         Pick one box with highest confidence
            # todo: slightly different from the original paper
            else:
                max_confidence = torch.tensor(0.).to(device)  # init
                if use_float64:
                    max_confidence = max_confidence.double()

                for b in range(B):
                    confidence = boxes[b][-1]

                    if use_float64:
                        confidence = confidence.double()

                    if confidence > max_confidence:
                        max_confidence = confidence
                        select = torch.tensor(b).to(device)

            # classification loss
            # copy the selected box info to Q
            Q[i, j, :5] = boxes[select]  # bbox (pos + confidence)
            Q[i, j, 5:] = P[i, j, -C:]                 # class probabilities
    return Q


def calc_loss_single(P, G, use_float64=False):
    """
    Compute multi-part loss function on a Single instance, for a Single bbox.
    :param P: Prediction tensor S*S*(5+C) with SINGLE bounding box per grid cell.
    :param G: GroundTruth tensor S*S*5

    :return: loss
    """
    if P.size(0) != S or P.size(1) != S or P.size(2) != 5+C:
        print(" Prediction tensor size is ", P.size())
        raise Exception("Tensor size not match")

    if G.size(0) != S or G.size(1) != S or G.size(2) != 5:
        print(" GroundTruth tensor size is ", G.size())
        raise Exception("Tensor size not match")
    
    loss = torch.zeros(1)  # init
    if use_float64:
        loss = loss.double()

    for i in range(S):
        for j in range(S):
            # case 1: grid cell HAS object
            if len(G[i, j, :].nonzero()) > 1:
                # localization
                loss = loss + LAMBDA_COORD * (torch.pow(P[i, j, 0] - G[i, j, 0], 2) + torch.pow(P[i, j, 1] - G[i, j, 1], 2))

                loss = loss + LAMBDA_COORD * (torch.pow(torch.sqrt(torch.abs(P[i, j, 2])) - torch.sqrt(torch.abs(G[i, j,2])), 2) \
                        + torch.pow(torch.sqrt(torch.abs(P[i, j, 3])) - torch.sqrt(torch.abs(G[i, j, 3])), 2))  # org
                # loss = loss + LAMBDA_COORD * (torch.sqrt(torch.abs(P[i, j, 2] - G[i, j, 2])) +
                #                               torch.sqrt(torch.abs(P[i, j, 3] - G[i, j, 3])))  # ZZ

                loss = loss + torch.pow(P[i, j, 4]-1, 2)   # Ground truth confidence is constant 1

                # classification
                true_cls = G[i, j, -1].type(torch.int64)
                true_cls_vec = torch.zeros(C)
                true_cls_vec[true_cls] = torch.tensor(1)
                pred_cls_vec = P[i, j, -C:]

                if use_float64:
                    pred_cls_vec = pred_cls_vec.double()
                    true_cls_vec = true_cls_vec.double()
                loss = loss + torch.sum(torch.pow(pred_cls_vec - true_cls_vec, 2))

            # case 2: grid cell NO object
            # classification
            else:
                loss = loss + LAMBDA_NOOBJ * torch.pow(P[i, j, 4]-0, 2)  # Ground truth confidence is constant 0
    return loss


def calc_loss(P_batch, G_batch, device=torch.device("cpu"), use_float64=False):
    """
    Compute multi-part loss function on a Batch.
    :param P_batch: Model Output.
                    Prediction tensor batch N*S*S*(5*B+C) with MULTIPLE bounding box per grid cell.
    :param G_batch:  GroundTruth tensor batch N*S*S*5
    :return:
    """
    if P_batch.size(0) != G_batch.size(0):
        raise Exception("Batch size does not match.")

    if len(P_batch.size()) != 4 or len(G_batch.size()) != 4:
        raise Exception("Input or Ground truth is not a Batch. ")

    total_loss = torch.tensor(0.0)
    if use_float64:
        total_loss.double()

    for i in range(P_batch.size(0)):
        P = P_batch[i]
        G = G_batch[i]
        Q = predict_one_bbox(P, G, device, use_float64)      # predict ONE bbox for each Grid cell
        total_loss = total_loss + calc_loss_single(Q, G, use_float64)   # compute Loss for one instance

    total_loss = total_loss / P_batch.size(0)
    return total_loss


if __name__ == "__main__":
    labels = read_labels('000001.txt')
    G = labels2tensor(labels)

    # Feed model
    yolo_model = build_darknet()

    # Data
    X = torch.randn(20, 3, 448, 448)                 # image batch (random)
    Y = torch.clamp(torch.randn(20, 7, 7, 5), 0, 1)  # label batch (random)

    # X.requires_grad = True
    # Y.requires_grad = True

    # Prediction
    Y_out = yolo_model(X)
    # Y_pred = [predict_one_bbox(Y_out[i].clone(), Y[i].clone()) for i in range(20)]
    # Y_pred = torch.stack(Y_pred)

    # Loss
    total_loss = calc_loss(Y_out, Y)
    print('total loss = ', total_loss)

    # Optimizer
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(yolo_model.parameters(), lr=learning_rate)

    # Training
    for t in range(30):
        # forward pass
        Y_out = yolo_model(X)

        # compute loss
        loss = calc_loss(Y_out.clone(), Y.clone())
        print('\nEpoch = ', t, 'Loss = ', loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Done.')




