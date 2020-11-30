import random
import torch
import numpy as np
import re
import logging

# basic method reading data from file
def loadFile(path):
    with open(path, "r", encoding="utf8") as df:
        data = df.readlines()
    return data

# basic method writing data into file
def saveFile(path, dataList):
    with open(path, "w", encoding="utf8") as df:
        for line in dataList:
            df.write(line+"\n")

# get file's line number quickly
def getFileLength(path):
     return len(["" for l in open(path, "r", encoding="utf8")])

def loadNpy(path):
    print("load npy:", path)
    return np.load(path)

def saveNpy(data, path):
    np.save(path, data)
    print("saved", path)

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速

def printConfig(config):
    d = config.__dict__
    for var in d:
        p = re.compile("__.*__")
        m = p.search(var)
        if m == None:
            print("config.%s=%s" % (var, d[var]))

def init_logger(config):
    logger = logging.getLogger(config.fileName)
    logger.setLevel(level=logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    handler = logging.FileHandler(config.log_dir + "log_" + config.fileName + "_" + config.dataset + "_Model" + str(
        config.modelName) + "_" + str(config.startTimeSpan) + ".txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(ch)
    return logger

def init_norm_Vector(relinit, entinit, embedding_size):
    zero_vec = [0.1 for i in range(embedding_size)]
    lstent = [zero_vec]
    lstrel = [zero_vec]
    with open(relinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstrel.append(tmp)
    with open(entinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstent.append(tmp)
    assert embedding_size % len(lstent[0]) == 0
    return np.array(lstent, dtype=np.float32), np.array(lstrel, dtype=np.float32)
