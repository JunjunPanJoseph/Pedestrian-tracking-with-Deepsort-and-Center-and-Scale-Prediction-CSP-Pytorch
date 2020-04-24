class Config(object):
    def __init__(self):
        self.use_cuda = True
        self.num_gpu = 2
        self.csp_checkpoint = './checkpoints/CSP_epoch_6.pth'
        self.deepsort_checkpoint = './checkpoints/Deepsort_ckpt.t7'
        self.test_batch_size = 40

        self.train_size = (576, 768)
        self.test_size = (576, 768)
        self.stride = 4
        self.radius = 2
        self.epoch_size = 20
        #self.epoch_size = 300
        self.score_thres = 0.2
        self.nms_thres = 0.3
