import os
import numpy as np
import cv2
import pickle as pkl
import random
from utils import *

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def draw_single(img_name, label_dir='./', out_dir='./', show_flag=False):
    """
    Draw bounding boxes of a SINGLE image.
    Automatically find labels based on image names.

    Note: Labels share the same name as images, using YOLO format.
          e.g. Image = 000001.jpg
               Label = 000001.txt
                        format = <class> <x> <y> <w> <h>
                        11 0.344192634561 0.611 0.416430594901 0.262
                        14 0.509915014164 0.51 0.974504249292 0.972


    :param img_name:    single image name / path + name
    :param label_dir:   the corresponding label directory
    :param out_dir:     declare output directory, which will be created if not exist.
    :param show_flag:   display if True.
    :return:
    """
    # Read image
    file_name = img_name.split('/')[-1].split('.')[0]

    img = cv2.imread(img_name)
    height, width = img.shape[:2]

    # Read label
    labels = read_labels(os.path.join(label_dir, file_name + '.txt'))

    # Color
    colors = pkl.load(open('pallete', 'rb'))
    font = cv2.FONT_HERSHEY_SIMPLEX
    m = 10

    # Draw box + class
    for l in labels:
        cls = classes[int(l[0])]
        upper_left_x = int((l[1] - l[3] / 2) * width)
        upper_left_y = int((l[2] - l[4] / 2) * height)
        bottom_right_x = int((l[1] + l[3] / 2) * width)
        bottom_right_y = int((l[2] + l[4] / 2) * height)

        color = random.choice(colors)
        cv2.rectangle(img, (upper_left_x, upper_left_y), (bottom_right_x, bottom_right_y), color, 3)

        if len(l) > 5:
            # has confidence score
            cv2.putText(img, cls + ' ' + str(l[5]), (upper_left_x - m, upper_left_y - m), font, 0.8, color, 2)
        else:
            # no confidence score
            cv2.putText(img, cls, (upper_left_x - m, upper_left_y - m), font, 0.8, color, 2)

    cv2.imwrite(os.path.join(out_dir, 'det_' + file_name + '.png'), img)

    if show_flag:
        cv2.imshow(file_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def draw(img_dir, label_dir, out_dir, show_flag=False):
    """
    Draw bounding boxes of MULTIPLE images.

        Note: Labels share the same name as images, using YOLO format.
          e.g. Image = 000001.jpg
               Label = 000001.txt
                        format = <class> <x> <y> <w> <h>
                        11 0.344192634561 0.611 0.416430594901 0.262
                        14 0.509915014164 0.51 0.974504249292 0.972

    :param img_dir:     directory of images OR 
                        list of image names
    :param label_dir:   directory of labels
    :param out_dir:     declare output directory, which will be created if not exist.
    :param show_flag:   display if True.
    :return:
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('"{}" is created.'.format(out_dir))
    else:
        print('"{}" exists.'.format(out_dir))


    # Image sources    
    if isinstance(img_dir, list):  # from list of image names
        img_list = img_dir
    else:                          # from directory of images
        img_list = os.listdir(img_dir)
        img_list = [os.path.join(img_dir, elem) for elem in img_list]

    for img_name in img_list:
        draw_single(img_name, label_dir, out_dir, show_flag)  # core


def visualize(y_out_epoch, img_name_epoch, image_list, out_dir, conf_threshold=0.1):
    """
    Visualize bbox a batch/epoch of images
    :param y_out_epoch:         N * S * S * (B * 5+C) Tensor
    :param img_name_epoch:      list of image name
    :param image_list:          list of path + image_name
    :param out_dir:             output to be stored here
    :param conf_threshold:      filter out bbox with small confidence
    :return:
    """
    assert y_out_epoch.size(0) == len(img_name_epoch)

    # convert to image coordinate [0,1]
    # #### Do ONLY once !!!
    Tensors = [convert_coord_cell2img(y_out_epoch[i]) for i in range(y_out_epoch.size(0))]

    # loop over each image
    for k in range(y_out_epoch.size(0)):
        T = y_out_epoch[k]
        img_name = img_name_epoch[k]
        res = []  # results to be write to .txt

        # loop over each grid cell
        for i in range(S):
            for j in range(S):
                _, cls = torch.max(T[i, j, :][-C:], 0)

                best_conf = 0
                for b in range(B):
                    bbox = [cls.item()]
                    bbox = bbox + T[i, j, 5*b: 5*b+5].tolist()

                    if b == 0:
                        best_bbox = bbox

                    # for each grid cell, select the box with highest confidence score
                    if T[i, j, 5*b+4] > best_conf:
                        best_bbox = bbox

                # filter out bbox with small confidence
                if best_bbox[-1] > conf_threshold:
                    res.append(best_bbox)

        # write to file
        with open(os.path.join(out_dir, img_name.split('.')[0] + '.txt'), 'w') as f:
            for r in res:
                for index in range(len(r)):
                    if index == 0:
                        f.write("%d " % r[index])
                    else:
                        f.write("%.4f " % r[index])
                f.write("\n")

    # draw box
    draw(image_list, out_dir, out_dir)


if __name__ == "__main__":
    # #  Single
    # img_name = '000001.jpg'
    img_name = '2008_000008.jpg'
    draw_single(img_name, show_flag=True)  # automatically find label based on image name

    # # Multiple

    img_dir = '/Users/erica/Workspace/my-yolo-implementation/data/image'
    label_dir = '/Users/erica/Workspace/my-yolo-implementation/data/label'
    out_dir = '/Users/erica/Workspace/my-yolo-implementation/det'
    
    draw(img_dir, label_dir, out_dir, show_flag=True)


    print('Done.')