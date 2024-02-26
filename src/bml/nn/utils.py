import torch.nn.functional as F

def easy_pad(t, tgt_shape=(200, 200, 52)):
    if t.shape[-3:] == tgt_shape:
        return t
    diff = [tgt - st for (tgt, st) in zip(tgt_shape[::-1], t.shape[::-1])]
    pad_arg = list()
    for d in diff:
        q, r = divmod(d, 2)
        pad_arg.append(q + r)
        pad_arg.append(q)
    return F.pad(t, pad_arg, 'constant', 0)
