import torch
import numpy as np
from dataset import *

IMG_WIDTH = 448
IMG_HEIGHT = 448
S = 7   # number of grid cell is S*S
B = 2   # number of bbox for each grid cell
C = 20  # number of classes
# C = 1  # debug!!!

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_labels(label_file):
    """
    Read labels from files
    Note:
    use YOLO author's format <class> <x> <y> <w> <h>
    :param label_file: a .txt file
    :return: labels: [list]
    """
    with open(label_file, 'r') as f:
        lines = f.readlines()
        labels = []
        for l in lines:
            l = l.split()
            l = [float(elem) for elem in l]
            labels.append(l)
    return labels


def labels2tensor(labels):
    """
    Build Groundtruth tensor S*S*5.
    :param labels: list of labels with bounding box classification and position for each image.
    :return: T: Groundtruth tensor S*S*5.
                format <x> <y> <w> <h> <class name>
    """
    T = torch.zeros(S, S, 5)  # init

    gcell_size = 1. / S
    for label in labels:  # mark labels
        cls = label[0]
        x = label[1]
        y = label[2]
        w = label[3]
        h = label[4]
        # Be aware: row are x-axis image coordinate, in 2nd dimension of Tensor

        T[int(y/gcell_size), int(x/gcell_size), 0] = x
        T[int(y/gcell_size), int(x/gcell_size), 1] = y
        T[int(y/gcell_size), int(x/gcell_size), 2] = w
        T[int(y/gcell_size), int(x/gcell_size), 3] = h
        T[int(y/gcell_size), int(x/gcell_size), 4] = cls

        '''
        # w,h already related to whole image, no action required
        # normalize x,y to grid cell offset
        x = (x - int(x/gcell_size) * gcell_size) / gcell_size
        y = (y - int(y/gcell_size) * gcell_size) / gcell_size
        '''
        T[int(y / gcell_size), int(x / gcell_size)] = torch.tensor([x, y, w, h, cls])

    return T


def convert_coord_cell2img(T):
    """
    Convert x, y from grid cell offset to image coordinate [0, 1]
    :param T: Prediction tensor S*S*(B*5+C)
              format < <x> <y> <w> <h> <confidence> > * B + <cls_prob>
    :return: T: converted
    Note:
            Clone input argument!!!
    Example:
            >> T = torch.zeros(3,3,5)
            >> T2 = convert_coord_cell2img(T.clone())
    """
    # T.requires_grad = True  # need backprop

    if T.size(0) != S or T.size(1) != S or T.size(2) != 5*B+C:
        raise Exception("Tensor size not match")

    # Be aware: row are x-axis image coordinate, in 2nd dimension of Tensor
    for b in range(B):
        # cells with object, that is, <confidence> != 0
        cells_index = T[:, :, b*5-1].nonzero()
        for i in range(cells_index.size(0)):
            (m, n) = cells_index[i]
            m = int(m)
            n = int(n)
            # grid cell offset to normalized image coordinates
            T[m, n, b*5] = n*(1. / S) + T[m, n, b*5].clone() *(1. / S)  # x
            T[m, n, b*5+1] = m*(1. / S) + T[m, n, b*5+1].clone() *(1. / S)  # y
    return T

'''
def calc_IOU(box_1, box_2):
    """
    compute IOU between two bounding boxes
    :param box_1: (x, y, w, h) image coordinates in [0, 1]
    :param box_2: (x, y, w, h) image coordinates in [0, 1]
    :return:
    """
    x_min_1 = torch.clamp(torch.Tensor((box_1[0] - box_1[2] / 2,)), 0, 1)
    x_max_1 = torch.clamp(torch.Tensor((box_1[0] + box_1[2] / 2,)), 0, 1)
    y_min_1 = torch.clamp(torch.Tensor((box_1[1] - box_1[3] / 2,)), 0, 1)
    y_max_1 = torch.clamp(torch.Tensor((box_1[1] + box_1[3] / 2,)), 0, 1)

    x_min_2 = torch.clamp(torch.Tensor((box_2[0] - box_2[2] / 2,)), 0, 1)
    x_max_2 = torch.clamp(torch.Tensor((box_2[0] + box_2[2] / 2,)), 0, 1)
    y_min_2 = torch.clamp(torch.Tensor((box_2[1] - box_2[3] / 2,)), 0, 1)
    y_max_2 = torch.clamp(torch.Tensor((box_2[1] + box_2[3] / 2,)), 0, 1)

    overlap_width = max(min(x_max_1, x_max_2) - max(x_min_1, x_min_2), 0)
    overlap_height = max(min(y_max_1, y_max_2) - max(y_min_1, y_min_2), 0)

    overlap_area = overlap_width * overlap_height
    union_area = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) \
                 + (x_max_2 - x_min_2) * (y_max_2 - y_min_2) \
                 - overlap_area
    intersection_over_union = overlap_area / union_area
    return intersection_over_union
'''

# def calc_IOU(box_1, box_2, device=torch.device('cpu'), use_float64=False):
def calc_IOU(box_1, box_2, device=torch.device('cpu'), use_float64=False):
    """
    Tensor version of calc_IOU()
    compute IOU between two bounding boxes
    :param box_1: Detection x, y, w, h image coordinates in [0, 1]
    :param box_2: GroundTruth x, y, w, h image coordinates in [0, 1]
    :return:
    """
    '''
    x_min_1 = torch.clamp((box_1[0] - box_1[2] / 2), 0, 1).to(device)
    x_max_1 = torch.clamp((box_1[0] + box_1[2] / 2), 0, 1).to(device)
    y_min_1 = torch.clamp((box_1[1] - box_1[3] / 2), 0, 1).to(device)
    y_max_1 = torch.clamp((box_1[1] + box_1[3] / 2), 0, 1).to(device)
    '''

    x_min_1 = torch.clamp((abs(box_1[0]) - abs(box_1[2]) / 2), 0, 1).to(device)
    x_max_1 = torch.clamp((abs(box_1[0]) + abs(box_1[2]) / 2), 0, 1).to(device)
    y_min_1 = torch.clamp((abs(box_1[1]) - abs(box_1[3]) / 2), 0, 1).to(device)
    y_max_1 = torch.clamp((abs(box_1[1]) + abs(box_1[3]) / 2), 0, 1).to(device)

    x_min_2 = torch.clamp((box_2[0] - box_2[2] / 2), 0, 1).to(device)
    x_max_2 = torch.clamp((box_2[0] + box_2[2] / 2), 0, 1).to(device)
    y_min_2 = torch.clamp((box_2[1] - box_2[3] / 2), 0, 1).to(device)
    y_max_2 = torch.clamp((box_2[1] + box_2[3] / 2), 0, 1).to(device)


    # z = torch.tensor(0, dtype=torch.float).to(device)
    z = torch.tensor(0.).to(device)
    if use_float64:
        z = z.double()

    a = torch.min(x_max_1, x_max_2)
    b = torch.max(x_min_1, x_min_2)
    c = torch.min(y_max_1, y_max_2)
    d = torch.max(y_min_1, y_min_2)

    overlap_width = torch.max(a-b, z)
    overlap_height = torch.max(c-d, z)
    overlap_area = overlap_width * overlap_height

    union_area = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) \
                 + (x_max_2 - x_min_2) * (y_max_2 - y_min_2) \
                 - overlap_area
    intersection_over_union = overlap_area / union_area
    return intersection_over_union


def prediction2detection(Tensors, Images, conf_threshold=0.1):
    """
    In the Evaluation stage, summarize the detection results in terms of object class.
    :param Tensors:
                Prediction tensors N*S*S*(B*5+C) with multiple bounding box per grid cell.
                format < <x> <y> <w> <h> <confidence> > * B + <cls_prob>

    :return: Detections: A dictionary contains bounding boxes for each object class over all images.
                format - [key]  class_id
                         [value]
                                list of bounding boxes
                                format <image_name> <x> <y> <w> <h> <confidence>
    Note:
            Model output x, y are in grid cell offset coordinates.
            Must convert to image coordinates before use.

    Update:
            06/27/2109: For each grid cell, only the bbox with highest confidence score will be used.

    """
    if not isinstance(Images, list):
        raise Exception("Expect a list of Image Names.")
    if Tensors.size(0) != len(Images):
        raise Exception("Number of tensors does not number of images.")
    if Tensors[0].size(0) != S or Tensors[0].size(1) != S:
        raise Exception("Tensor size not match")
    if Tensors[0].size(2) != 5*B+C and Tensors[0].size(2) != 5+C:
        raise Exception("Tensor size not match")

    # convert to image coordinate [0,1]
    # #### Do ONLY once !!!
    Tensors = [convert_coord_cell2img(Tensors[i]) for i in range(Tensors.size(0))]

    # init
    Detections = dict()
    for c in range(C):
        Detections[c] = []

    for k in range(len(Tensors)):
        T = Tensors[k]
        img_name = Images[k]
        for i in range(S):
            for j in range(S):
                _, cls = torch.max(T[i, j, :][-C:], 0)

                best_conf = 0  # record the highest confidence
                for b in range(B):
                    bbox = (img_name,)
                    bbox = bbox + tuple(T[i, j, 5*b: 5*b + 5])

                    if b == 0:
                        best_bbox = bbox

                    if T[i, j, 5*b+4] > best_conf:
                        best_bbox = bbox
                # bbox with highest confidence score will be used.
                # Detections[cls.item()].append(best_bbox)
                if best_bbox[-1] > conf_threshold:
                    Detections[cls.item()].append(best_bbox)
    return Detections


def ground_truth_detection(label_list):
    """
    In the Evaluation stage, summarize the Ground Truth in terms of object class.
    :param label_list: a list of label file names
    :return: Detections: A dictionary contains Ground Truth bounding boxes
                        for each object class over all images.
                        format - [key]  class_id
                                 [value]
                                    list of bounding boxes
                                    format <image_name> <x> <y> <w> <h>
    """
    Detections = dict()
    for c in range(C):
        Detections[c] = []

    for k in range(len(label_list)):
        img_name = label_list[k].split('/')[-1].replace('.txt', '')
        labels = read_labels(label_list[k])
        for label in labels:
            Detections[int(label[0])].append((img_name,) + tuple(label[1:]))
    return Detections


def evaluate_IOU(Detections, Ground_truth, device=torch.device('cpu'), use_float64=False):
    """
    Compute IOU over all images.
    :param Detections: A dictionary contains bounding boxes
                       for each object class over all images.
                        format - [key]  class_id
                                 [value]
                                    list of bounding boxes
                                    <image_name> <x> <y> <w> <h> <confidence>
    :param Ground_truth: A dictionary contains bounding boxes
                        for each object class over all images.
                        format - [key]  class_id
                                 [value]
                                    list of bounding boxes
                                    <image_name> <x> <y> <w> <h>
    :return:
           Results: A dictionary contains bounding boxes
                       for each object class over all images.
                    format - [key]  class_id
                             [value]  
                                list of bounding boxes 
                                <image_name> <x> <y> <w> <h> <confidence> <IOU>
    """
    Results = {}
    for c in range(C):
        Det = Detections[c]  # predicted detection
        GT = Ground_truth[c]  # ground truth
        Results[c] = []
        for det in Det:
            img_ground_truth = list(filter(lambda x: x[0] == det[0].split('.')[0], GT))
            if len(img_ground_truth) > 0:
                '''
                if use_float64:
                    inter_over_unions = [calc_IOU(det[1:5], torch.tensor(gt[1:5]).double(), device, use_float64) for gt in img_ground_truth]
                else:
                    inter_over_unions = [calc_IOU(det[1:5], torch.tensor(gt[1:5]), device, use_float64) for gt in img_ground_truth]
                '''
                inter_over_unions = []
                for gt in img_ground_truth:
                    if use_float64:
                        curr_iou = calc_IOU(det[1:5], torch.tensor(gt[1:5]).double(), device, use_float64)
                    else:
                        curr_iou = calc_IOU(det[1:5], torch.tensor(gt[1:5]), device, use_float64)
                    inter_over_unions.append(curr_iou.item())

                iou = max(inter_over_unions)
                img_ground_truth.pop(np.argmax(inter_over_unions))  # remove matched ground truth
            else:
                iou = 0.0
            Results[c].append(list(det) + [iou])
    return Results


def evaluate_TP_FP(Results, threshold):

    """
    Computer TP or FP based on threshold.
    :param Results: A dictionary contains bounding boxes
                        for each object class over all images.
                    format - [key]  class_id
                             [value]
                                list of bounding boxes
                                <image_name> <x> <y> <w> <h> <confidence> <IOU>
    :param threshold:
                    detection is TP if IOU > threshold, otherwise FP
                        for each object class over all images.

    :return: Results: A dictionary contains bounding boxes
                        for each object class over all images.
                        format - [key]  class_id
                                 [value]
                                    list of bounding boxes
                                    <image_name> <x> <y> <w> <h> <confidence> <IOU> <TP> <FP>
    """
    if not 0 <= threshold <= 1:
        raise Exception("IOU threshold should be in [0, 1]")
    
    for c in range(len(Results)):    
        for i in range(len(Results[c])):
            if Results[c][i][-1] > threshold:  # IOU > threshold
                Results[c][i] += [1, 0]  # TP
            else:
                Results[c][i] += [0, 1]  # FP
    return Results
    

def evaluate_precision_recall(Results, threshold, all_ground_truths):
    """
    Compute Precision and Recall based on threshold.
    :param Results:  A dictionary contains bounding boxes
                        for each object class over all images.
                        format - [key]  class_id
                                 [value]   
                                    list of bounding boxes
                                    <image_name> <x> <y> <w> <h> <confidence> <IOU> <TP> <FP>
    :param threshold:
                     detection is TP if IOU > threshold, otherwise FP

    :return: Results: A dictionary contains bounding boxes
                        for each object class over all images.
                        format - [key]  class_id
                                 [value]
                                    list of bounding boxes
                                    <image_name> <x> <y> <w> <h> <confidence> <IOU> <TP> <FP>
                                    <Acc TP> <Acc FP> <precision> <recall>

            Acc_tp_all_cls:         accumulated TP for all classes [List]
            Acc_fp_all_cls:         accumulated FP for all classes [List]
            Precisions_all_cls:     Precisions for all classes [List]
            Recalls_all_cls:        Recalls for all classes [List]

    """
    Acc_tp_all_cls = []
    Acc_fp_all_cls = []
    Precisions_all_cls = []   # PR curve points for each class
    Recalls_all_cls = []

    acc_tp = 0.0  # accumulated detection
    acc_fp = 0.0  # accumulated detection
    eps = 1e-10   # prevent division over zero
    
    # sort by confidence
    def take_confidence(elem):
        return elem[5]

    for c in range(C):    
        Results[c].sort(key=take_confidence, reverse=True)  # order detections by their confidences

    # all groundtruth
    num_all_groundtruth = 0
    for c in range(len(Results)):
        for i in range(len(Results[c])):  
            num_all_groundtruth += len(all_ground_truths[c])  
        
    # compute Accumulated TP, Accumulated FP, Precision, Recall
    for c in range(len(Results)):
        precisions = []
        recalls = []

        acc_tp = 0 # ZZ
        acc_fp = 0 # ZZ
        for i in range(len(Results[c])):
            res = Results[c][i]
            acc_tp += res[-2]
            acc_fp += res[-1]
            precision = acc_tp / (acc_tp + acc_fp + eps)
            recall = 0 if len(all_ground_truths[c]) == 0 else (acc_tp / len(all_ground_truths[c]))  

            # record
            Results[c][i] += [acc_tp, acc_fp, precision, recall]
            precisions.append(precision)  # for Precision-Recall curve
            recalls.append(recall)

        Acc_tp_all_cls.append(acc_tp)
        Acc_fp_all_cls.append(acc_fp)
        Precisions_all_cls.append(precisions)
        Recalls_all_cls.append(recalls)

    return Results, Acc_tp_all_cls, Acc_fp_all_cls, Precisions_all_cls, Recalls_all_cls


def calc_average_precision(p, r, show_flag=False):
    """
    Calculate Average Precision by interpolating PR-curve.

    Note: Interpolation performed in all points.

    :param p: Precision points [list]
    :param r: Recall points    [list]
    :param show_flag: plot if TRUE  [boolean]

    :return: ap:        Average Precision
             p_interp:  interpolated precision
    """
    assert len(p) == len(r), "Equal number of Precision and Recall points."
    ap = 0.0
    # add starting point (r, p) = (0, 1)
    p = [1] + p
    r = [0] + r
    p_interp = [p[0]]

    for i in range(len(p)-1):
        interp = max(p[i:])
        ap += (r[i+1] - r[i]) * interp
        p_interp.append(interp)

    if show_flag:
        plt.plot(r, p)
        plt.step(r, p_interp)
        plt.legend(['Precision', 'Interpolated precision'], loc='upper right')
        plt.show()
    return ap, p_interp


def calc_mean_average_precision(p_all_cls, r_all_cls):
    """
    Calc mAP.
    :param p_all_cls:   Precisions for all classes [list]
    :param r_all_cls:   Recalls for all classes [list]
    :return: mAP
    """
    assert len(p_all_cls) == len(r_all_cls), "lengths of Lists should be equal."

    mAP = 0
    num_class = len(p_all_cls)

    non_empty_list = [x for x in p_all_cls if x != []]
    num_class_effective = len(non_empty_list)

    for i in range(num_class):
        p = p_all_cls[i]
        r = r_all_cls[i]
        ap, _ = calc_average_precision(p, r)
        mAP += ap

    mAP = mAP / num_class_effective if num_class_effective else 0
    return mAP


if __name__ == "__main__":
    voc2012 = VOC('/Users/erica/Dataset/Pascal/2012_train_short.txt', IMG_WIDTH, IMG_HEIGHT)
    det = ground_truth_detection(voc2012.label_list)
    print('Done.')


