class Config:
    data_path = '.\\data\\cmn_jianti.txt'
    model_save_dir = '.\\tf_ckpt'
    tboard_save_dir = '.\\tf_log'
    test_size = 0.2
    data_split_seed = 20190507


    num_heads = 8
    num_blocks = 6
    # vocab
    input_vocab_size = 50
    label_vocab_size = 50
    # embedding size
    max_length = 50
    hidden_units = 512
    dropout_rate = 0.2

    epochs = 50
    train_batch_size = 64
    test_batch_size = 64
    evaluate_batch_interval = 100
    save_epoch_interval = 1
config = Config()