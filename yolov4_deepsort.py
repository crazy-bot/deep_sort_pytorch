import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

#from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
################### import for V4 ##################

from models import *
from utils.utils import *


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        #self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        #self.class_names = self.detector.class_names

        # Set up model
        model = Darknet(args.config_path, img_size=args.img_size)
        model.load_weights(args.weights_path)
        print('model path: ' + args.weights_path)
        if use_cuda:
            model.cuda()
            print("using cuda model")
        model.eval()
        self.model = model
        self.class_names = load_classes(args.class_path)  # Extracts class labels from file
        self.Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "6_track_v4.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            resized_img, img = resize_img(ori_im, self.args.img_size)
            #print((resized_img.size(),type(resized_img)))
            #cv2.imshow('img',img)
            input_imgs = Variable(resized_img.type(self.Tensor))
            #im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                detections = self.model(input_imgs)
                # print(detections)
                alldetections = non_max_suppression(detections, 3, self.args.conf_thres, self.args.nms_thres)
            detection = alldetections[0]
            # The amount of padding that was added
            pad_x = max(img.shape[0] - img.shape[1], 0) * ( self.args.img_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * ( self.args.img_size / max(img.shape))
            # Image height and width after padding is removed
            unpad_h =  self.args.img_size - pad_y
            unpad_w =  self.args.img_size - pad_x
            bbox_xywh, cls_confs, cls_ids = [],[],[]
            if detection is not None:
                for j, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detection):
                    #cls_pred = min(cls_pred, max(unique_labels))
                    # print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                    # Rescale coordinates to original dimensions
                    box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
                    box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]))
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))
                    # x2 = int(x1 + box_w)
                    # y2 = int(y1 + box_h)
                    bbox_xywh.append([x1+box_w/2,y1+box_h/2,box_w,box_h])
                    cls_confs.append(cls_conf)
                    cls_ids.append(cls_pred)

            bbox_xywh = np.asarray(bbox_xywh)
            cls_confs = np.asarray(cls_confs)
            cls_ids = np.asarray(cls_ids)
            # do detection
            #bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # select person class
            #mask = cls_ids == 0

            #bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            #bbox_xywh[:, 3:] *= 1.2
            #cls_conf = cls_conf[mask]

            # do tracking
            if(len(bbox_xywh) > 0):
                outputs = self.deepsort.update(bbox_xywh, cls_confs, ori_im)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                    results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            # self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
            #                  .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default='/data/Guha/construction/Videos/6.mp4')
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true",default=False)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="/data/Guha/construction/output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")


    ################# added for V4 ###################
    parser.add_argument('--config_path', type=str, default='configs/mydata_yolov4.cfg', help='path to model config file')
    parser.add_argument("--data_config_path", type=str, default="configs/mydata.data", help="path to data config file")
    parser.add_argument('--weights_path', type=str, default="/data/Guha/construction/checkpoints/pytorch/mydata_best.weights",
                        help='path to weights file')
    parser.add_argument('--class_path', type=str, default='configs/mydata.names', help='path to class label file')
    parser.add_argument('--conf_thres', type=float, default=0.9, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.5, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--use_cuda', type=bool, default=False, help='whether to use cuda if available')
    parser.add_argument("--output", default='/data/Guha/construction/output/6_track_v4.mp4')
    parser.add_argument("--input", default='/data/Guha/construction/Videos/6.mp4')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
