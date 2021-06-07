import torch
import argparse

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--feat", default='features/padded.pth', type=str)
args = parser.parse_args()


feat_path = args.feat
features = torch.load(feat_path)
qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]

scores = qf.mm(gf.t())
flat_scores = torch.flatten(scores)
print(torch.max(flat_scores))
print(torch.min(flat_scores))

print(scores.shape)
res = scores.topk(5, dim=1)[1][:,0]
print(scores.topk(5, dim=1)[1])
print(res)
print(gl[res])
top1correct = gl[res].eq(ql).sum().item()
print("Acc top1: {:.3f}".format(top1correct/ql.size(0)))


def calculate_topk(scores, k):
    print("--------------------------")
    print(f"Computing top{k} accuracy")
    print("--------------------------")
    sim = scores.topk(k+1, dim=1)[0][1:]
    res = scores.topk(k+1, dim=1)[1][1:]  # Quitamos el primero ya que es la misma imagen
    print(f"SHAPE SIM: {sim.shape}")
    print(f"Resultados top {k}")
    print(gl[res])
    topkcorrect = 0
    for idx, (cls, topk) in enumerate(zip(ql, gl[res])):
        cls = cls.item()
        topk = topk.numpy()
        # (cls == topk).all()
        if cls in topk:
            topkcorrect += 1

    print("--------------------------")
    result = topkcorrect/scores.shape[0]
    print(f"Acc top{k}: {result:.3f}")

    return result


calculate_topk(scores, 1)
calculate_topk(scores, 3)
calculate_topk(scores, 5)
calculate_topk(scores, 10)








