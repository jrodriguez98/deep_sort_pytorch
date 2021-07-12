import os
import os.path as osp
import logging
import argparse
from pathlib import Path

from utils.log import get_logger
from rcnn_deepsort import VideoTracker
from utils.parser import get_config

import motmetrics as mm
mm.lap.default_solver = 'lap'
from utils.evaluation import Evaluator

def mkdir_if_missing(dir):
    os.makedirs(dir, exist_ok=True)


def get_tracker_type(args):
    tracker_type = None
    print(f"ARGS")
    print(args)
    print(f"CONFIG TRACKER")
    print(args.config_tracker)
    if "deep_sort" in args.config_tracker:
        tracker_type = "deep_sort"
    elif "sort" in args.config_tracker:
        tracker_type = "sort"
    else:
        raise ValueError(f"Tracker desconocido: {args.config_tracker}")

    return tracker_type

def main(data_root='', seqs=('',), args=""):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    model_name = args.MODEL_NAME
    data_type = 'mot'
    analyse_every_frames = args.frame_interval
    dataset_name = data_root.split(sep='/')[-1]
    tracker_type = get_tracker_type(args)
    result_root = os.path.join("mot_results", model_name, dataset_name)
    mkdir_if_missing(result_root)

    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_tracker)

    args.save_path = result_root

    # run tracking
    accs = []
    for seq in seqs:
        logger.info('start seq: {}'.format(seq))
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        seq_root = os.path.join(data_root, seq)
        video_root = os.path.join(seq_root, "video")
        video_path = os.path.join(video_root, os.listdir(video_root)[0])

        logger.info(f"Result filename: {result_filename}")
        logger.info(f'Frame interval: {analyse_every_frames}')
        if not os.path.exists(result_filename):
            with VideoTracker(cfg, args, video_path, result_filename) as vdo_trk:
                vdo_trk.run()
        else:
            print(f"Result file {result_filename} already exists. Skipping processing")


        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_global.xlsx'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("MODEL_NAME", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/faster_rcnn_17.yaml")
    parser.add_argument("--config_tracker", type=str, default="./configs/deep_sort_distorn.yaml")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    seqs_str = '''seguimiento1       
                  seguimiento2
                  gospandi
                  gospandi2
                  gospandi3
                  jilavur
                  khusab
                  khusab2
                  lolezar
                  move
                  '''

    datasets_str = '''vehicles_1   
                  vehicles_2 
                  vehicles_3 
                  vehicles_5 
                  vehicles_6 
                  vehicles_10 
                  '''
    data_root = 'data/dataset/'

    seqs = [seq.strip() for seq in seqs_str.split()]
    datasets = [dataset.strip() for dataset in datasets_str.split()]

    for dataset in datasets:
        dataset_dir = os.path.join(data_root, dataset)
        print(dataset_dir)
        args.fps = 30/int(dataset_dir.split(sep="/")[-1].split(sep="_")[-1])
        main(data_root=dataset_dir, seqs=seqs, args=args)
