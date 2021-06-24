from .custom_sort import Sort

__all__ = ['Sort', 'build_sort_tracker']


def build_sort_tracker(cfg):
    print(cfg)
    return Sort(max_age=cfg.MAX_AGE, min_hits=cfg.N_INIT, iou_threshold=cfg.IOU_THRES)
