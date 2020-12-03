import progressbar as pb
from torch.autograd import Variable
import torch.nn.functional as nnf
import torch
import numpy as np

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.BCE_loss = torch.nn.BCEWithLogitsLoss()
        self.CE_loss = torch.nn.CrossEntropyLoss()
        self.KL_criterion = torch.nn.KLDivLoss(reduction='batchmean')  # reduction='none'
        self.KL_criterion2 = torch.nn.KLDivLoss(reduction='none')  # reduction='none'
        self.MSE_criterion = torch.nn.MSELoss()
        self.L1_criterion = torch.nn.L1Loss()
        self.cos_sim = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

    def train(self, s_model, t_model, train_ds, opt, epoch_idx):
        s_model.train(mode=True)
        t_model.train(mode=True)
        pbar = pb.ProgressBar(widgets=["epoch %d|" % (epoch_idx + 1),
                                       pb.Percentage(), pb.Bar(), pb.ETA()], maxval=train_ds.nbatches)
        pbar.start()
        totalLoss, num_batches_completed = 0, 0
        total_loss_vec = np.zeros(3)
        for singlebatch in train_ds.getBatch():
            loss, loss_vec = self.train_batch(s_model, t_model, epoch_idx, singlebatch, opt)
            totalLoss += loss.data
            total_loss_vec += loss_vec
            pbar.update(num_batches_completed)
            num_batches_completed += 1
        pbar.finish()
        print("KD, BCE, BCE2，", total_loss_vec)
        return totalLoss.item()

    def train_batch(self, s_model, t_model, epoch_idx, singlebatch, opt):
        config = self.config
        opt.zero_grad()
        e1_batch, r_batch, e2_batch = [], [], []
        for item in singlebatch:
            e1, e2, r = item[:3]
            e1_batch.append([e1])
            r_batch.append([r])
            e2_batch.append([e2])
        entity_varb = Variable(torch.LongTensor(np.array(e1_batch, dtype=np.int32)))
        relation_varb = Variable(torch.LongTensor(np.array(r_batch, dtype=np.int32)))
        target_varb = Variable(torch.LongTensor(np.array(e2_batch, dtype=np.int32)))
        if config.cuda == True:
            entity_varb = entity_varb.cuda()
            relation_varb = relation_varb.cuda()
            target_varb = target_varb.cuda()

        rand_index = torch.randint(low=0, high=self.config.entityNum,
                                   size=[len(entity_varb), self.config.negTriple_num])
        candidate = torch.cat([target_varb, rand_index.cuda()], dim=1)
        stucand_score = s_model.forward(entity_varb, relation_varb, candidate)

        label = torch.zeros_like(stucand_score).float()
        label[:, 0] = 1
        with torch.no_grad(): # test1  test2#
            total_score = s_model.forward(entity_varb, relation_varb).clone().detach()
            topk_score, topk_candidate = torch.topk(total_score, k=self.config.kd_topk, dim=1)
            topk_label = torch.where(topk_candidate == target_varb, torch.ones_like(topk_candidate),torch.zeros_like(topk_candidate)).float()
            del (total_score, topk_score)
        stutopk_score = s_model.forward(entity_varb, relation_varb, topk_candidate).double()

        KD_loss, teatopk_score = t_model.loss(entity_varb, relation_varb, topk_candidate, stutopk_score, epoch_idx)

        BCE_loss = self.BCE_loss(input=stucand_score, target=label)
        BCE_loss2 = self.BCE_loss(input=teatopk_score.float(), target=topk_label)

        loss = BCE_loss + BCE_loss2 + self.config.kd_ratio * KD_loss
        loss.backward()
        opt.step()
        s_model.constraint()
        del (entity_varb, relation_varb, target_varb)
        loss_vec = np.array([KD_loss.detach().cpu().numpy().item(),
                             BCE_loss.detach().cpu().numpy().item(),
                             BCE_loss2.detach().cpu().numpy().item()])
        return loss, loss_vec

    def preload(self, t_model, train_ds):
        t_model.eval()
        pbar = pb.ProgressBar(widgets=["preload|",
                                       pb.Percentage(), pb.Bar(), pb.ETA()], maxval=int(train_ds.data_size//5)+1)
        pbar.start()
        num_batches_completed = 0
        pair_set = set([str(h)+"_"+str(r) for h,t,r in train_ds.xs])
        recordDict = {pair:"" for pair in pair_set}
        for singlebatch in train_ds.getBatch(5):
            entity_varb, relation_varb, score_vecs, save_ids = self.preload_batch(t_model, singlebatch)
            for h,r,pred,ents in zip(entity_varb, relation_varb, score_vecs, save_ids):
                key = str(h[0])+"_"+str(r[0])
                if recordDict[key] == "":
                    recordDict[key] = [pred,ents]
            pbar.update(num_batches_completed)
            num_batches_completed += 1
        pbar.finish()
        return recordDict

    def preload_batch(self, t_model, singlebatch):
        sbatch = singlebatch
        config = self.config
        e1_batch, r_batch, e2_batch = [], [], []
        for item in sbatch:
            e1, e2, r = item[:3]
            e1_batch.append([e1])
            r_batch.append([r])
            e2_batch.append([e2])
        entity_varb = Variable(torch.LongTensor(np.array(e1_batch, dtype=np.int32)))
        relation_varb = Variable(torch.LongTensor(np.array(r_batch, dtype=np.int32)))
        target_varb = Variable(torch.LongTensor(np.array(e2_batch, dtype=np.int32)))
        if config.cuda == True:
            entity_varb = entity_varb.cuda()
            relation_varb = relation_varb.cuda()
            target_varb = target_varb.cuda()
        with torch.no_grad():
            score_vecs, save_ids = t_model.preforward(entity_varb, relation_varb)
            score_vecs = score_vecs.cpu().numpy()
            save_ids = save_ids.cpu().numpy()
            entity_varb = entity_varb.cpu().numpy()
            relation_varb = relation_varb.cpu().numpy()
        #print(entity_varb.shape, relation_varb.shape, score_vecs.shape, save_ids.shape)
        return entity_varb, relation_varb, score_vecs, save_ids


class Tester(object):
    def __init__(self, config):
        self.config = config

    def test(self, model, valid_ds, mode):
        model.eval()
        test_batch = valid_ds.nbatches  # 20
        config = self.config
        pbar = pb.ProgressBar(widgets=["eval %s|" % mode, pb.Percentage(), pb.Bar(), pb.ETA()], maxval=test_batch)
        pbar.start()
        num_batches_completed = 0
        metricsVec = []
        metricsVecH, metricsVecT = [], []
        for singlebatch in valid_ds.getBatch():
            e1, e2, rel, pred1, pred2 = self.test_batch(model, singlebatch)
            torch.cuda.empty_cache()

            batch_size = len(singlebatch)
            for i in range(batch_size):
                num1 = e1[i, 0].item()
                num2 = e2[i, 0].item()
                num_rel = rel[i, 0].item()
                filter1 = valid_ds.getFilter((num1, num2, num_rel), mode="test")
                filter2 = valid_ds.getFilter((num2, num1, num_rel + config.relationNum), mode="test")
                target_value1 = pred1[i, num2].item()
                target_value2 = pred2[i, num1].item()
                pred1[i][list(filter1)] = -1e35  # 0.0
                pred2[i][list(filter2)] = -1e35  # 0.0
                # write base the saved values
                # pred1[i][e2[i]] = target_value1
                # pred2[i][e1[i]] = target_value2

                # metricsH, topk_scores, predict_rank, predict_level = mrr_mr_hitk_new(pred1[i], num2, sort_mode)
                target_rank = torch.sum(pred1[i] >= target_value1).cpu().numpy() + 1
                metricsH = [float(1 / target_rank), float(target_rank), int(target_rank <= 10), int(target_rank <= 3),
                            int(target_rank <= 1)]
                metricsVec.append(metricsH)
                metricsVecH.append(metricsH)

                # metricsT, topk_scores, predict_rank2, predict_level = mrr_mr_hitk_new(pred2[i], num1, sort_mode)
                target_rank = torch.sum(pred2[i] >= target_value2).cpu().numpy() + 1
                metricsT = [float(1 / target_rank), float(target_rank), int(target_rank <= 10), int(target_rank <= 3),
                            int(target_rank <= 1)]
                metricsVec.append(metricsT)
                metricsVecT.append(metricsT)

            num_batches_completed += 1
            if num_batches_completed >= test_batch:
                break
            pbar.update(num_batches_completed)
        pbar.finish()

        mrr, mr, hit10, hit3, hit1 = np.array(metricsVecH).mean(axis=0)
        print("#" * 10)
        print('H MRR:', round(np.mean(1. / np.array(metricsVecH)[:, 1]), 4), ' MR:', round(mr, 2),
              'Hit@10:', round(hit10 * 100, 2),
              'Hit@3:', round(hit3 * 100, 2),
              'Hit@1:', round(hit1 * 100, 2))

        mrr, mr, hit10, hit3, hit1 = np.array(metricsVecT).mean(axis=0)
        print('T MRR:', round(np.mean(1. / np.array(metricsVecT)[:, 1]), 4), ' MR:', round(mr, 2),
              'Hit@10:', round(hit10 * 100, 2),
              'Hit@3:', round(hit3 * 100, 2),
              'Hit@1:', round(hit1 * 100, 2))

        mrr, mr, hit10, hit3, hit1 = np.array(metricsVec).mean(axis=0)

        print('MRR:', round(np.mean(1. / np.array(metricsVec)[:, 1]), 4), ' MR:', round(mr, 2),
              'Hit@10:', round(hit10 * 100, 2),
              'Hit@3:', round(hit3 * 100, 2),
              'Hit@1:', round(hit1 * 100, 2))

        hit10 = round(hit10 * 100, 2)
        hit1 = round(hit1 * 100, 2)
        mrr = round(mrr * 100, 2)
        if "valid" in mode:
            scoreList = config.Top5Dev + [mrr]
            config.Top5Dev = sorted(scoreList, reverse=True)[:len(config.Top5Dev)]
            print('Top 5 Dev Score: {0}', config.Top5Dev)
        elif "test" in mode:
            scoreList = config.Top5Test + [mrr]
            config.Top5Test = sorted(scoreList, reverse=True)[:len(config.Top5Test)]
            print('Top 5 Test Score: {0}', config.Top5Test)
        return mrr,hit10,hit1


    def test_batch(self, model, singlebatch):
        sbatch = singlebatch
        config = self.config
        e1_batch, r_batch, e2_batch, r2_batch = [], [], [], []
        for item in sbatch:
            e1, e2, r = item[:3]
            e1_batch.append(e1)
            r_batch.append(r)
            e2_batch.append(e2)
            r2_batch.append(r + config.relationNum)
        batch_size = len(singlebatch)
        n_ent = config.entityNum
        batch_s = torch.LongTensor(np.array(e1_batch, dtype=np.int32))
        batch_r = torch.LongTensor(np.array(r_batch, dtype=np.int32))
        batch_t = torch.LongTensor(np.array(e2_batch, dtype=np.int32))
        batch_r2 = torch.LongTensor(np.array(r2_batch, dtype=np.int32))

        # rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, 1))
        # src_var = Variable(batch_s.unsqueeze(1).expand(batch_size, 1))
        # dst_var = Variable(batch_t.unsqueeze(1).expand(batch_size, 1))
        # rel2_var = Variable(batch_r2.unsqueeze(1).expand(batch_size, 1))
        rel_var = Variable(batch_r.unsqueeze(1))
        src_var = Variable(batch_s.unsqueeze(1))
        dst_var = Variable(batch_t.unsqueeze(1))
        rel2_var = Variable(batch_r2.unsqueeze(1))
        if config.cuda == True:
            rel_var = rel_var.cuda()
            src_var = src_var.cuda()
            dst_var = dst_var.cuda()
            rel2_var = rel2_var.cuda()
        pred1 = model.score(src_var, rel_var)
        pred2 = model.score(dst_var, rel2_var)
        e1, e2, rel = src_var.data, dst_var.data, rel_var.data
        pred1, pred2 = pred1.data, pred2.data
        del (rel_var, src_var, dst_var, rel2_var)
        return e1, e2, rel, pred1, pred2
