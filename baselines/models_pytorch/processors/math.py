import torch

def floyd(heads, max_len):
    INF = 1e8
    inf = torch.ones_like(heads, device=heads.device, dtype=heads.dtype) * INF
    # replace 0 with infinite
    dist = torch.where(heads==0, inf.long(), heads.long())
    for k in range(max_len):
        for i in range(max_len):
            for j in range(max_len):
                if dist[i][k] != INF and dist[k][j] != INF and dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    zero = torch.zeros_like(heads, device=heads.device).long()
    dist = torch.where(dist==INF, zero, dist)
    return dist

def compute_distance(heads, mask, debug=False):
    if debug:
        torch.set_printoptions(profile="full")

    lengths = [sum(m) for m in mask]
    dists = []
    logger.info("Start computing distance ...")
    # for each sentence
    for i in range(len(heads)):
        if i % 1 == 0:
            print ("%d..."%i, end="")
        if debug:
            print ("heads:\n", heads[i])
            print ("mask:\n", mask[i])
            print ("lengths:\n", lengths[i])
        dist = floyd(heads[i], lengths[i])
        dists.append(dist)
        if debug:
            print ("dist:\n", dist)
    return dists