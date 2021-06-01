import numpy
import torch
import numpy as np
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-08)

features = torch.load("features.pth")
qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]

scores = qf.mm(gf.t())
print(scores.shape)
res = scores.topk(5, dim=1)[1][:,0]
print(scores.topk(5, dim=1)[1])
print(res)
print(gl[res])
top1correct = gl[res].eq(ql).sum().item()
print(f"QL SIZE 0: {ql.size(0)}")
print("Acc top1: {:.3f}".format(top1correct/ql.size(0)))

def calculate_topk(scores, k):
    res = scores.topk(k, dim=1)[1]

    topkcorrect = 0
    for cls, topk in zip(ql, gl[res]):
        if cls in topk:
            topkcorrect += 1

    result = topkcorrect/scores.shape[0]
    print(f"Acc top{k}: {result:.3f}")

    return result

calculate_topk(scores, 3)

#calculate_topk(scores, 5)



"""def calculate_top_k(qf, ql, gf, gl, k):

    print(f"qf shape: {qf.shape}")
    print(f"gf shape: {gf.shape}")

    print(gf[0].shape)
    distances = []
    num_correct = 0
    for i in range(qf.shape[0]):
        for j in range(gf.shape[0]):
            dist = qf[i].dot(gf[j])
            distances.append(dist)

        dist_sorted = torch.argsort(torch.Tensor(distances))  # ArgSort distances
        top_k = dist_sorted[:k]  # Get indexes
        #print(f"Indexes: {top_k}")
        top_k = gl[top_k]  # Get classes

        #print(f"TOP K: {top_k}")
        #print(f"Class: {ql[i]}")
        distances = []
        if ql[i] in top_k:
            print("CORRECT")
            print(top_k)
            print(ql[i])
            num_correct += 1

    return num_correct/ql.size(0)

print(calculate_top_k(qf, ql, gf, gl, 10))"""




