import numpy as np


class AvgMeter(object):
    def __init__(self, num=40):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []
        self.num = num

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val.item())

    def show(self):
        return np.mean(self.losses[np.maximum(len(self.losses) - self.num, 0):])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict


def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x + 1e-6) ** -1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    tpdiag = np.diag(hist)
    tp = tpdiag[1]
    sum_a1 = hist.sum(axis=1)
    TPFN = sum_a1[1]
    sum_a0 = hist.sum(axis=0)
    TPFP = sum_a0[1]

    recall = tp / (TPFN + np.finfo(np.float32).eps)
    precision = tp / (TPFP + np.finfo(np.float32).eps)
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1


def cm2score(confusion_matrix):
    hist = confusion_matrix
    tpdiag = np.diag(hist)
    tp = tpdiag[1]
    sum_a1 = hist.sum(axis=1)
    TPFN = sum_a1[1]
    sum_a0 = hist.sum(axis=0)
    TPFP = sum_a0[1]

    acc = tpdiag.sum() / (hist.sum() + np.finfo(np.float32).eps)
    recall = tp / (TPFN + np.finfo(np.float32).eps)
    precision = tp / (TPFP + np.finfo(np.float32).eps)
    F1 = 2 * ((recall * precision) / (recall + precision + np.finfo(np.float32).eps))

    tpiou = tpdiag[1] / (hist.sum() - tpdiag[0] + np.finfo(np.float32).eps)
    cls_iou = {"iou": tpiou}

    cls_precision = {"precision": precision}
    cls_recall = {"recall": recall}
    cls_F1 = {"F1": F1}
    score_dict = {"acc": acc}

    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    def __fast_hist(label_gt, label_pred):
        mask = (label_gt >= 0) & (label_gt < num_classes)

        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask], minlength=num_classes ** 2)
        hist = hist.reshape(num_classes, num_classes)
        return hist

    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix
