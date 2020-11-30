import numpy as np
import torch
import copy, math
from random import randint
from source.Utils import *
from collections import defaultdict

class KG_Loader(object):
    def __init__(self, path, ifprint = True):
        self.data_path = path
        self.ifprint = ifprint
        # load idDict
        self.entityNum, self.entityDict = self.load_Item2idDict(self.data_path + "entity2id.txt")
        self.relationNum, self.relationDict = self.load_Item2idDict(self.data_path + "relation2id.txt")

        # load triples
        self.trainList, self.trainNum = self.load_Triples(self.data_path + "train.txt")
        self.validList, self.validNum = self.load_Triples(self.data_path + "valid.txt")
        self.testList, self.testNum = self.load_Triples(self.data_path + "test.txt")

        self.trainList2, self.trainFilter = self.process_Triples(self.trainList) # build extend list + build filter dict
        self.testList2, self.testFilter = self.process_Triples(self.validList + self.testList, self.trainFilter)
        if "WN" in path or "FB" in path:
            self.rel_probVec, self.rel_typeVec = self.compute_rel_info() # get relation type

        if ifprint:
            print("trainNum", self.trainNum, "validNum", self.validNum, "testNum", self.testNum)
            print("entityNum", self.entityNum, "relationNum", self.relationNum)
            print("trainHTNum", len(self.trainList2), "testHTNum", len(self.testList2))
            print("trainFilterNum", len(self.trainFilter), "testFilterNum", len(self.testFilter))

    def process_Triples(self, tripleList, filter_dict = None):
        newList = copy.deepcopy(tripleList)
        if filter_dict is None:
            filter_dict = defaultdict(lambda: defaultdict(lambda: set()))
        else:
            filter_dict = copy.deepcopy(filter_dict)
        for h,t,r in tripleList:

            r2 = r + self.relationNum
            newList.append([t, h, r + self.relationNum])
            filter_dict[r][h].add(t)
            filter_dict[r2][t].add(h)
        return newList, filter_dict

    def getFilter(self, triple, mode="train"):
        h, t, r = triple
        if mode == "train":
            return self.trainFilter[r][h]
        elif mode == "test":
            return self.testFilter[r][h]

    def compute_rel_info(self):
        bern_prob = torch.zeros(self.relationNum * 2)
        rel_type = torch.zeros(self.relationNum * 2)
        for r in range(1, self.relationNum):
            r2 = r + self.relationNum
            rel_dict = self.trainFilter[r]
            rel_dict2 = self.trainFilter[r2]

            tph, hpt = 0, 0
            if len(rel_dict) > 0:
                tph = sum(len(tails) for tails in rel_dict.values()) / len(rel_dict)
            if len(rel_dict2) > 0:
                hpt = sum(len(heads) for heads in rel_dict2.values()) / len(rel_dict2)
            bern_prob[r] = tph / (tph + hpt)

            if tph < 1.5 and hpt < 1.5: type = 0 # 1-1
            elif tph >= 1.5 and hpt < 1.5: type = 1 # 1-n
            elif tph < 1.5 and hpt >= 1.5: type = 2 # n-1
            elif tph >= 1.5 and hpt >= 1.5: type = 3 # n-n
            rel_type[r] = type
        return bern_prob, rel_type

    # load from item2id.txt,
    def load_Item2idDict(self, file, sp="\t"):
        list = {}
        data = loadFile(file)
        list["null"] = 0
        for line in data:
            items = line.strip().split(sp)
            if len(items) == 2:
                item, id = items
                list[item] = int(id) + 1  # note: id+1, set 0 as null
        return len(list), list

    # load from train/test/valid.txt
    def load_Triples(self, file, rank = "htr", sp="\t"):
        list = []
        data = loadFile(file)
        for line in data:
            triple = line.strip().split(sp)
            if(len(triple)<3):
                continue
            if rank == "hrt":
                h, r, t = triple
            elif rank == "htr":
                h, t, r = triple
            if h not in self.entityDict.keys() or t not in self.entityDict.keys():
                continue
            if r not in self.relationDict.keys():
                continue
            list.append(tuple([self.entityDict[h],self.entityDict[t],self.relationDict[r]]))
        return list, len(list)

class KG_DataSet(object):
    def __init__(self, xs, loader, mode, config):
        self.xs = xs
        self.mode = mode
        self.config = config
        self.loader = loader
        self.recordDict = None
        self.data_size = len(xs)
        self.batch_size = 16
        if mode == "train":
            self.batch_size = config.batch_size
        elif mode == "test" or mode == "valid":
            self.batch_size = config.test_batch_size
        self.dataType = config.dataType
        self.nbatches = math.ceil(self.data_size / float(self.batch_size))
        return

    def getFilter(self, triple, mode = "train"):
        return self.loader.getFilter(triple, mode)

    def getBatch(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.mode == "train":
            rand_idx = np.random.permutation(self.data_size)
        else:
            rand_idx = range(self.data_size)
        start = 0
        while start < self.data_size:
            end = min(start + batch_size, self.data_size)
            Sbatch = [self.xs[i] for i in rand_idx[start:end]]
            yield Sbatch
            start = end

