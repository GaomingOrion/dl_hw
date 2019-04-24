class Config:
    data_dir = '..\\data'
    model_save_dir = '.\\tf_ckpt'
    tboard_save_dir = '.\\tf_log'
    train_size = 33402
    test_size = 13068

    image_shape = (64, 192)
    nr_channel = 3
    seq_length = 24
    rnn_hidden_units = 256
    rnn_layer_num = 1
    rnn_keep_prob = 0.5
    class_num = 11

    epochs = 20
    train_batch_size = 64
    test_batch_size = 64
    evaluate_batch_interval = 100
    save_epoch_interval = 1

    @property
    def input_shape(self):
        return self.image_shape + (3,)

config = Config()