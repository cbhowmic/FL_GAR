class Arguments():
    def __init__(self):
        self.dataset = 'HAR'
        self.network = 'MLP'
        self.total_rows = 10299
        self.clients = 30
        self.rounds = 1000
        self.local_rounds = 5
        self.local_batches = 10
        self.lr = 0.01
        self.C = 0.9   # 0.9
        self.drop_rate = 0  # 0.1
        self.torch_seed = 0
        self.log_interval = 10
        self.iid = 'iid'
        self.split_size = int(self.total_rows / self.clients)
        self.samples = self.split_size / self.total_rows
        self.use_cuda = True
        self.save_model = True
        self.aggregate = ['average','median', 'GM', 'TM', 'SM', 'krum', 'multi_krum']  # 'median' #'average' #'SM'  'TSM' 'krum','bulyan', 'TM', 'GM' multi_krum 'krum', 'GM', 'medoid', 'SM'
        # ['krum', 'median' 'average', 'GM', 'SM']
        self.temperature = 10  # for soft-medoid and trimmed soft-medoid
        self.trim_size = 2
        self.attack = True
        self.attack_type = 'label_flipping'     # 'byzantine'     # 'label_flipping'   # 'sign_flipping'
        self.num_attacked = 3
        self.save_data = True