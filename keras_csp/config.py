class Config(object):
    def __init__(self):
        self.gpu_ids = '0'
        self.onegpu = 16
        self.num_epochs = 100
        self.add_epoch = 0
        self.iter_per_epoch = 2000
        self.init_lr = 1e-4
        self.alpha = 0.999

        # setting for network architechture
        self.network = 'resnet50'  # or 'mobilenet'
        self.point = 'center'  # or 'top', 'bottom
        self.scale = 'hw'  # or 'w', 'hw'
        self.num_scale = 2  # 1 for height (or width) prediction, 2 for height+width prediction
        self.offset = True  # append offset prediction or not
        self.down = 4  # downsampling rate of the feature map for detection
        self.radius = 2  # surrounding areas of positives for the scale map

        # setting for data augmentation
        # self.use_horizontal_flips = True
        self.use_horizontal_flips = False
        self.brightness = (0.5, 2, 0.5)
        self.size_train = (512, 512)
        self.size_test = (512, 512)

        # image channel-wise mean to subtract, the order is BGR
        # self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_channel_mean = [0, 0, 0]


