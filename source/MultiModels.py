import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_
import numpy as np
from source.preModel_Dict import *
import source.Equations as Equations
import math, os


class BasicFunction(nn.Module):
    def __init__(self, config, hidden_dim):
        super(BasicFunction, self).__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.rank = self.hidden_dim # config.embedding_dim

        self.ent_embed = nn.Embedding(config.entityNum, self.hidden_dim).cuda()
        self.rel_embed = nn.Embedding(config.relationNum * 2, self.hidden_dim).cuda() # config.embedding_dim
        self.head_bias_embed = nn.Embedding(config.entityNum, 1).cuda()
        self.tail_bias_embed = nn.Embedding(config.entityNum, 1).cuda()
        self.hidden_drop = torch.nn.Dropout(config.dropout_value)

        self.init_size = 0.001
        self.data_type = torch.float #float
        self.ent_embed.weight.data = self.init_size * torch.randn((self.config.entityNum, self.rank),dtype=self.data_type)
        self.rel_embed.weight.data = self.init_size * torch.randn((self.config.relationNum * 2, self.rank),dtype=self.data_type)
        self.head_bias_embed.weight.data = torch.zeros((self.config.entityNum, 1), dtype=self.data_type)
        self.tail_bias_embed.weight.data = torch.zeros((self.config.entityNum, 1), dtype=self.data_type)

    def forward(self, src, rel, candidate = None):
        src_embeded = self.ent_embed(src)
        src_embeded = self.hidden_drop(src_embeded)
        if candidate is None:
            dst_embeded = self.ent_embed.weight
            tail_bias = self.tail_bias_embed.weight.squeeze(1).unsqueeze(0)
        else:
            dst_embeded = self.ent_embed(candidate)
            tail_bias = self.tail_bias_embed(candidate).squeeze(2)
        head_bias = self.head_bias_embed(src).squeeze(2)
        score = self.score_function(src_embeded, rel, dst_embeded)
        return score + head_bias + tail_bias

    def score_function(self, head, rel_ids, tail):
        return

class TransE_Function(BasicFunction):
    def __init__(self, config, hidden_dim):
        super(TransE_Function, self).__init__(config, hidden_dim)
        self.function_name = "TransE"

    def score_function(self, head, rel_ids, tail):
        relation = self.rel_embed(rel_ids)
        query = head + relation
        score_vec = (query - tail) ** 2
        score = -torch.sum(score_vec, dim=2)
        return score

class DistE_Function(BasicFunction):
    def __init__(self, config, hidden_dim):
        super(DistE_Function, self).__init__(config, hidden_dim)
        self.function_name = "DistE"

    def score_function(self, head, rel_ids, tail):
        relation = self.rel_embed(rel_ids)
        query = head * relation
        score_vec = (query - tail) ** 2
        score = -torch.sum(score_vec, dim=2)
        return score

class RotE_Function(BasicFunction):
    def __init__(self, config, hidden_dim):
        super(RotE_Function, self).__init__(config, hidden_dim)
        self.rot_embed = nn.Embedding(config.relationNum * 2, config.embedding_dim)
        #xavier_uniform_(self.rot_embed.weight.data, gain=1)
        self.rot_embed.weight.data = self.init_size * torch.randn((self.config.relationNum * 2, self.rank),dtype=self.data_type)
        self.function_name = "RotE"

    def score_function(self, head, rel_ids, tail):
        relation = self.rel_embed(rel_ids)
        rel_diag = self.rot_embed(rel_ids)
        query = Equations.euc_rotations(rel_diag, head).unsqueeze(1)
        query = query + relation
        score_vec = (query - tail) ** 2
        score = -torch.sum(score_vec, dim=2)
        return score

class RefE_Function(BasicFunction):
    def __init__(self, config, hidden_dim):
        super(RefE_Function, self).__init__(config, hidden_dim)
        self.ref_embed = nn.Embedding(config.relationNum * 2, config.embedding_dim)
        #xavier_uniform_(self.ref_embed.weight.data, gain=1)
        self.ref_embed.weight.data = self.init_size * torch.randn((self.config.relationNum * 2, self.rank),dtype=self.data_type)
        self.function_name = "RefE"

    def score_function(self, head, rel_ids, tail):
        relation = self.rel_embed(rel_ids)
        rel_diag = self.ref_embed(rel_ids)
        query = Equations.euc_reflection(rel_diag, head).unsqueeze(1)
        query = query + relation
        score_vec = (query - tail) ** 2
        score = -torch.sum(score_vec, dim=2)
        return score

class TransH_Function(BasicFunction):
    def __init__(self, config, hidden_dim):
        super(TransH_Function, self).__init__(config, hidden_dim)
        self.c = torch.nn.Parameter(Variable(torch.ones(1, dtype=self.data_type)))
        if self.config.cuda: self.c.cuda()
        self.function_name = "TransH"

    def score_function(self, head, rel_ids, tail):
        relation = self.rel_embed(rel_ids)
        c = torch.nn.functional.softplus(self.c)
        head = Equations.expmap0(head, c)
        relation = Equations.expmap0(relation, c)
        tail = Equations.expmap0(tail, c)
        query = Equations.mobius_add(head, relation, c)
        score = -Equations.hyp_distance(query, tail, c)**2
        score = score.squeeze(2)
        return score

class DistH_Function(BasicFunction):
    def __init__(self, config, hidden_dim):
        super(DistH_Function, self).__init__(config, hidden_dim)
        self.c = torch.nn.Parameter(Variable(torch.ones(1, dtype=self.data_type)))
        if self.config.cuda: self.c.cuda()
        self.function_name = "DistH"

    def score_function(self, head, rel_ids, tail):
        relation = self.rel_embed(rel_ids)
        c = torch.nn.functional.softplus(self.c)
        head = Equations.expmap0(head, c)
        relation = Equations.expmap0(relation, c)
        tail = Equations.expmap0(tail, c)
        query = head * relation
        score = -Equations.hyp_distance(query, tail, c)**2
        score = score.squeeze(2)
        return score

class RefH_Function(BasicFunction):
    def __init__(self, config, hidden_dim):
        super(RefH_Function, self).__init__(config, hidden_dim)
        self.c = torch.nn.Parameter(Variable(torch.ones(1, dtype=self.data_type)))
        if self.config.cuda: self.c.cuda()
        self.ref_embed = nn.Embedding(config.relationNum * 2, config.embedding_dim)
        #xavier_uniform_(self.ref_embed.weight.data, gain=1)
        self.ref_embed.weight.data = self.init_size * torch.randn((self.config.relationNum * 2, self.rank), dtype=self.data_type)
        self.function_name = "RefH"

    def score_function(self, head, rel_ids, tail):
        relation = self.rel_embed(rel_ids)
        rel_diag = self.ref_embed(rel_ids)
        c = torch.nn.functional.softplus(self.c)
        head = Equations.expmap0(head, c)
        relation = Equations.expmap0(relation, c)
        tail = Equations.expmap0(tail, c)
        head = Equations.euc_reflection(rel_diag, head).unsqueeze(1)
        query = Equations.mobius_add(head, relation, c)
        score = -Equations.hyp_distance(query, tail, c) ** 2
        score = score.squeeze(2)
        return score

class RotH_Function(BasicFunction):
    def __init__(self, config, hidden_dim):
        super(RotH_Function, self).__init__(config, hidden_dim)
        self.c = torch.nn.Parameter(Variable(torch.ones(1, dtype=self.data_type)))
        if self.config.cuda: self.c.cuda()
        self.rot_embed = nn.Embedding(config.relationNum * 2, self.rank)
        self.rot_trans_embed = nn.Embedding(config.relationNum * 2, self.rank)
        # xavier_uniform_(self.rot_embed.weight.data, gain=1)
        # xavier_uniform_(self.rot_trans_embed.weight.data, gain=1)
        self.rot_embed.weight.data = self.init_size * torch.randn((self.config.relationNum * 2, self.rank), dtype=self.data_type)
        self.rot_trans_embed.weight.data = self.init_size * torch.randn((self.config.relationNum * 2, self.rank), dtype=self.data_type)
        self.function_name = "RotH"

    def score_function(self, head, rel_ids, tail):
        relation = self.rel_embed(rel_ids)
        rel_diag = self.rot_embed(rel_ids)
        rel_trans = self.rot_trans_embed(rel_ids)
        c = torch.nn.functional.softplus(self.c)
        head = Equations.expmap0(head, c)
        relation = Equations.expmap0(relation, c)
        rel_trans = Equations.expmap0(rel_trans, c)
        tail = Equations.expmap0(tail, c)
        head = Equations.mobius_add(head, rel_trans, c)
        head = Equations.euc_rotations(rel_diag, head).unsqueeze(1)
        query = Equations.mobius_add(head, relation, c)
        score = -Equations.hyp_distance(query, tail, c) ** 2
        score = score.squeeze(2)
        return score

class teacherModels(nn.Module):
    def __init__(self, config, funcnamelist):
        super(teacherModels, self).__init__()
        self.config = config
        self.preload = True
        self.func_list = []
        for i in range(len(funcnamelist)):
            self.func_list.append(self.select_func(funcnamelist[i], i))
        self.function_num = len(self.func_list)

    def forward(self, src, rel, candidate=None):
        scores = []
        total_score = 0
        with torch.no_grad():
            for i in range(self.function_num):
                score_vec = self.func_list[i](src, rel, candidate).double().clone().detach()
                scores.append(score_vec)
                total_score += score_vec
        return scores

    def select_func(self, modelName, index):
        if self.preload:
            model_name = self.config.dataset + "_" + modelName
            self.__dict__["func" + str(index)] = torch.load(
                self.config.load_path + model_name +"_"+ modelDict[model_name][dimDict[str(self.config.teacher_dim)]])
        else:
            self.__dict__["func" + str(index)] = globals()[modelName + "_Function"](self.config, self.config.teacher_dim).cuda()
        return self.__dict__["func" + str(index)]

class SeniorModel(nn.Module):
    def __init__(self, config, tmodels):
        super(SeniorModel, self).__init__()
        self.config = config
        self.KL_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        self.KL_criterion2 = torch.nn.KLDivLoss(reduction='none')
        self.rel_gate_embed = nn.Embedding(config.relationNum * 2, 4)
        self.rel_gate_embed.weight.data.copy_(torch.ones([self.config.relationNum * 2, 4]).float())
        self.T = self.config.kd_temp  # 0.5 #1
        self.tmodels = tmodels

    def forward(self, src, rel, candidate):
        tea_scores = self.tmodels(src, rel, candidate)
        rel_gate = self.rel_gate_embed(rel.squeeze(1))
        rel_gate = torch.sigmoid(rel_gate).unsqueeze(1)
        seq_vecs = torch.stack(tea_scores, dim=2)
        rs_seq_vecs = seq_vecs * rel_gate.double()
        return rs_seq_vecs, seq_vecs

    def score(self, src, rel, candidate=None):
        rs_seq_vecs,_ = self.forward(src, rel, candidate)
        seq_vec = torch.sum(rs_seq_vecs, dim=2)  #
        return seq_vec

    def origin_score(self, src, rel, candidate=None):
        _, seq_vecs = self.forward(src, rel, candidate)
        seq_vec = torch.sum(seq_vecs, dim=2)  #
        return seq_vec

    def loss(self, entity, relation, candidate, stu_score, epoch_idx):
        rs_seq_vecs, seq_vecs = self.forward(entity, relation, candidate)
        teatopk_scores = [item.squeeze(-1) for item in torch.split(rs_seq_vecs, 1, dim=2)]
        stutopk_score = stu_score
        exp_radio = math.exp(epoch_idx // 5)
        diff_list = []
        for single_score in teatopk_scores:
            single_score = single_score.double().clone().detach()
            diff = self.KL_criterion2(nnf.log_softmax(single_score.float(), dim=1),
                                      nnf.softmax(stutopk_score.float(), dim=1) + 1e-7)
            diff = torch.sum(diff, dim=1)
            diff_list.append(diff)
        diff_vec = torch.softmax(-torch.stack(diff_list, dim=1) / exp_radio, dim=1)
        teadiff_score = torch.sum(torch.stack(teatopk_scores, dim=2) * diff_vec.unsqueeze(1).double(), dim=2)
        KD_loss = self.KL_criterion(nnf.log_softmax(stutopk_score.float() / self.T, dim=1),
                                    nnf.softmax(teadiff_score.float() / self.T, dim=1) + 1e-7) * self.T * self.T
        return KD_loss, teadiff_score


class JuniorModel(nn.Module):
    def __init__(self, config, funcname):
        super(JuniorModel, self).__init__()
        self.config = config
        self.func = globals()[funcname + "_Function"](self.config, self.config.embedding_dim).cuda()

        # model_name = self.config.dataset + "_" + funcname
        # self.func = torch.load(self.config.load_path + model_name + modelDict[model_name][dimDict[str(self.config.embedding_dim)]])

        self.BCE_loss = torch.nn.BCEWithLogitsLoss()
        self.KL_criterion = torch.nn.KLDivLoss(reduction='batchmean')

    def init(self): return

    def constraint(self): return

    def forward(self, src, rel, candidate=None):
        score_vec = self.func(src, rel, candidate)
        return score_vec

    def score(self, src, rel, candidate=None):
        return self.forward(src, rel, candidate)

    def loss(self, entity, relation, target):
        rand_index = torch.randint(low=0, high=self.config.entityNum, size=[len(entity), self.config.negTriple_num])
        if self.config.cuda: rand_index = rand_index.cuda()
        candidate = torch.cat([target, rand_index], dim=1)
        candidate_score = self.forward(entity, relation, candidate)
        # candidate_score = score.gather(dim=1, index = candidate)

        label = torch.zeros_like(candidate_score).float()
        label[:, 0] = 1
        loss = self.BCE_loss(input=candidate_score, target=label)
        return loss
