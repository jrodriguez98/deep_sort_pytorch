from .deep_sort import DeepSort


__all__ = ['DeepSort', 'build_deepsort_tracker']


def build_deepsort_tracker(cfg):

    return DeepSort(cfg.REID_CKPT, max_dist=cfg.MAX_DIST, min_confidence=cfg.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.NMS_MAX_OVERLAP, max_iou_distance=cfg.MAX_IOU_DISTANCE,
                    max_age=cfg.MAX_AGE, n_init=cfg.N_INIT, nn_budget=cfg.NN_BUDGET,
                    apply_pad=cfg.APPLY_PAD, use_cuda=True)
    









