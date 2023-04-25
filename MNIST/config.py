class Arguments():
    def __init__(self):
        self.dataset = 'MNIST'
        self.network = 'CNN'
        self.images = 60000
        self.clients = 20
        self.rounds = 400
        self.local_rounds = 2
        self.local_batches = 64
        self.lr = 0.01
        self.C = 0.9   # 0.9
        self.drop_rate = 0  # 0.1
        self.torch_seed = 0
        self.log_interval = 10
        self.iid = 'iid'
        self.split_size = int(self.images / self.clients)
        self.samples = self.split_size / self.images
        self.use_cuda = True
        self.save_model = True
        self.aggregate = ['bulyan']  # 'median' #'average' #'SM'  'TSM' 'krum','bulyan', 'TM', 'GM' multi_krum 'krum', 'GM', 'medoid', 'SM'
        # ['krum', 'median' 'average', 'GM', 'SM']
        self.temperature = 10  # for soft-medoid and trimmed soft-medoid
        self.trim_size = 5
        self.attack = False
        self.attack_type = 'byzantine'     # 'label_flipping'   # 'sign_flipping'
        self.num_attacked = 5
        self.save_data = True