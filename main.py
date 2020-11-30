from source.Utils import *
from source.KGloader import KG_DataSet, KG_Loader
from source.MultiModels import *
from source.Config import Config
from source.Trainer import Trainer, Tester
import itertools
import datetime
import torch
import re, os

config = Config()
config.dataset = "FB15k-237" #"FB15k-237" #"WN18RR"
config.modelName = "RotH"
config.learning_rate = 0.0005
config.kd_ratio = 0.5 #WN0.01  # FB0.5
config.kd_temp = 1
config.kd_topk = 500 #500
config.embedding_dim = 32
config.teacher_dim = 64
config.batch_size = 500 #128
config.negTriple_num = 255
config.test_period = 1 #5
config.test_batch_size = 20
config.load_path = "saveFunctions/"
config.save_path = "save/"
config.info = "" #Senior

loader = KG_Loader(config.data_path + config.dataset + "/")
config.entityNum = loader.entityNum
config.relationNum = loader.relationNum
config.startTimeSpan = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print("startTime:", config.startTimeSpan)

d = config.__dict__
for var in d:
    p = re.compile("__.*__")
    m = p.search(var)
    if m == None:
        print("config.%s=%s" % (var, d[var]))

train_ds = KG_DataSet(loader.trainList2, loader, "train", config)
valid_ds = KG_DataSet(loader.validList, loader, "valid", config)
test_ds = KG_DataSet(loader.testList, loader, "test", config)

s_model = JuniorModel(config, config.modelName)
s_model.init()
if config.cuda: s_model = s_model.cuda()

model_name_list = ["TransH", "DistH", "RotH", "RefH"]
tea_model = teacherModels(config, model_name_list)
if config.cuda: tea_model = tea_model.cuda()

t_model = SeniorModel(config, tea_model)
if config.cuda: t_model = t_model.cuda()

trainer = Trainer(config)
tester = Tester(config)
# tester.test(t_model, test_ds, "test")
# print("Teachers prevalid")

total_parameters = itertools.chain.from_iterable([t_model.parameters(), s_model.parameters()]) #
print("Senior Model")
for k,v in t_model.named_parameters():
    print(k, v.shape)
print("Junior Model")
for k,v in s_model.named_parameters():
    print(k, v.shape)

opt = torch.optim.Adam(total_parameters, lr=config.learning_rate, weight_decay=0.000)
best_metric = 0
for epoch_idx in range(config.trainTimes):
    loss = trainer.train(s_model, t_model, train_ds, opt, epoch_idx)
    print(loss, round(loss / train_ds.data_size, 4))
    if epoch_idx % config.test_period == 0:
        with torch.no_grad():
            metric = tester.test(s_model, valid_ds, "s_valid")
            mrr = metric[0]
            if mrr > best_metric and epoch_idx > 0:
                tester.test(t_model, valid_ds, "t_valid")
                best_metric = mrr
                best_epoch = epoch_idx
                torch.save(s_model,
                           config.save_path + config.info + "junior_model" + "_" + str(config.startTimeSpan) + ".pth")
                torch.save(t_model,
                           config.save_path + config.info + "senior_model" + "_" + str(config.startTimeSpan) + ".pth")
# evaluate
s_best_model = torch.load(config.save_path + config.info + "junior_model" + "_" + str(config.startTimeSpan) + ".pth")
tester.test(s_best_model, test_ds, "test")
