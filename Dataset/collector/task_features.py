import math, numpy as np

def make_task_features(op_type, shapes, dtype):
    total = 1
    for s in shapes:
        total *= int(np.prod(s))
    log_size = math.log10(total+1)
    return {"op_type": op_type, "log_size": log_size, "dtype": dtype}
