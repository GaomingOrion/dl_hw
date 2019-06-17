class Config:
    data_dir = '../data'
    model_save_dir = './tf_ckpt'
    tboard_save_dir = './tf_log'
    #vgg16_path = 'F:\\PycharmProjects\\Pre-trained\\vgg_16.ckpt'
    train_size = 50000
    test_size = 10000

    image_shape = (32, 32)
    nr_channel = 3
    seq_length = 12
    rnn_hidden_units = 256
    rnn_layer_num = 1
    rnn_keep_prob = 0.5
    class_num = 11

    epochs = 50
    train_batch_size = 64
    test_batch_size = 64
    evaluate_batch_interval = 200
    save_epoch_interval = 1

    @property
    def input_shape(self):
        return self.image_shape + (3,)

config = Config()