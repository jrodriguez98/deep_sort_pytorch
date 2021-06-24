import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
from PIL import Image

from detector import tensorflow_detection_tf2, load_model_in_gpu
from deep_sort import build_deepsort_tracker
from custom_sort import build_sort_tracker
from utils.draw import custom_draw
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from utils.utils_rcnn import VECTORES_INTERES, CATEGORIES, get_deep_format


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
        self.detect_fn = load_model_in_gpu(cfg.faster_rcnn.PATH)

        self.tracker_type = None
        if "DEEPSORT" in cfg.keys():
            self.mot_tracker = build_deepsort_tracker(cfg.DEEPSORT)
            self.tracker_type = "DEEPSORT"
        elif "SORT" in cfg.keys():
            self.mot_tracker = build_sort_tracker(cfg.SORT)
            self.tracker_type = "SORT"
        else:
            raise ValueError("MOT TRACKER HAS NOT BEEN GIVEN OR IS NOT SUPPORTED")

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
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
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
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            dict_images = {}
            dict_images[str(idx_frame)] = Image.fromarray(im)
            # do detection
            detection_result = tensorflow_detection_tf2(
                value_threshold='0.7',
                dict_images=dict_images,
                num_classes=7,
                vectores_interes=VECTORES_INTERES,
                categories=CATEGORIES,
                max_width_crop=1920,
                max_height_crop=1080,
                detect_fn=self.detect_fn
            )

            objects_detected = np.array(detection_result["objects_detected"], dtype=np.float)

            if self.tracker_type == "DEEPSORT":
                bbox_xywh, cls_conf, cls_ids = get_deep_format(np.copy(objects_detected))

                # do tracking
                trackers = self.mot_tracker.update(bbox_xywh, cls_conf, cls_ids, im)
            elif self.tracker_type == "SORT":
                trackers = self.mot_tracker.update(objects_detected)

            ori_im = custom_draw(ori_im, trackers)

            # draw boxes for visualization
            if len(trackers) > 0:
                bbox_tlwh = []
                bbox_xyxy = trackers[:, :4].astype(np.int32)
                identities = trackers[:, 4].astype(np.int32)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.mot_tracker._xyxy_to_tlwh(bb_xyxy))

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
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), objects_detected.shape[0], len(trackers)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="./videos/VBS3_20210318_seguimiento_2.mp4")
    parser.add_argument("--config_detection", type=str, default="./configs/faster_rcnn.yaml")
    parser.add_argument("--config_tracker", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_tracker)


    with VideoTracker(cfg, args, video_path=args.video_path) as vdo_trk:
        vdo_trk.run()
