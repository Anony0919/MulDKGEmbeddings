
class Config(object):
    def __init__(self):
        self.nei_size = 20
        self.dataset = "WN18RR"
        self.data_path = "data/"
        self.save_path = "save/"
        self.log_dir ="log/"
        self.trainFlag = True
        self.loadFlag = False
        self.cuda = True
        self.learning_rate = 0.001
        self.embedding_dim = 500
        self.trainTimes = 1000
        self.batch_size = 1000
        self.negTriple_num = 50
        self.dropout_value = 0
        self.test_batch_size = 10
        self.dataType = "pair"
        self.modelName = "RotH"
        self.test_period = 5
        self.margin = 9
        self.save_flag = False
        self.entityNum = 0
        self.relationNum = 0
        self.nbatches = 0
        self.Top5Dev = [0 for i in range(5)]
        self.Top5Test = self.Top5Dev.copy()
