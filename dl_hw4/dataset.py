from tqdm import tqdm
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from common import config

class Dataset:
    def __init__(self):
        inputs, outputs = self.load_data()
        SOURCE_CODES = ['<PAD>']
        TARGET_CODES = ['<PAD>', '<GO>', '<EOS>']
        self.encoder_vocab = self.get_vocab(inputs, init=SOURCE_CODES)
        self.decoder_vocab = self.get_vocab(outputs, init=TARGET_CODES)
        self.encoder_vocab_size = len(self.encoder_vocab[0].keys())
        self.decoder_vocab_size = len(self.decoder_vocab[0].keys())

        encoder_inputs = np.array([[self.encoder_vocab[0][word] for word in line] for line in inputs])
        decoder_inputs = np.array([[self.decoder_vocab[0]['<GO>']] + [self.decoder_vocab[0][word] for word in line] for line in
                          outputs])
        decoder_targets = np.array([[self.decoder_vocab[0][word] for word in line] + [self.decoder_vocab[0]['<EOS>']] for line in
                           outputs])

        tmp = np.array(train_test_split(encoder_inputs, decoder_inputs, decoder_targets,
                               test_size=config.test_size, random_state=config.data_split_seed))
        self.train_data, self.test_data = tmp[[0, 2, 4]], tmp[[1, 3, 5]]
        self.train_size, self.test_size = len(self.train_data[0]), len(self.test_data[0])
        del encoder_inputs, decoder_inputs, decoder_targets, tmp

    @staticmethod
    def load_data():
        with open(config.data_path, 'r', encoding='utf8') as f:
            data = f.readlines()
        inputs = []
        outputs = []
        for line in tqdm(data):
            [en, ch] = line.strip('\n').split('\t')
            inputs.append(en.replace(',', ' ,')[:-1].lower())
            outputs.append(ch[:-1])
        inputs = [en.split(' ') for en in inputs]
        outputs = [[char for char in jieba.cut(line) if char != ' '] for line in outputs]
        return inputs, outputs

    @staticmethod
    def get_vocab(data, init):
        vocab = init
        for line in tqdm(data):
            for word in line:
                if word not in vocab:
                    vocab.append(word)
        return ({vocab[i]: i for i in range(len(vocab))},
                {i: vocab[i] for i in range(len(vocab))})

    def one_epoch_generator_for_train(self):
        idx = list(range(self.train_size))
        np.random.shuffle(idx)
        begin = 0
        while begin < self.train_size:
            end = begin + config.train_batch_size
            en_input_batch = self.train_data[0][idx[begin:end]]
            de_input_batch = self.train_data[1][idx[begin:end]]
            de_target_batch = self.train_data[2][idx[begin:end]]
            max_en_len = max([len(line) for line in en_input_batch])
            max_de_len = max([len(line) for line in de_input_batch])
            en_input_batch = np.array([line + [0] * (max_en_len - len(line)) for line in en_input_batch])
            de_input_batch = np.array([line + [0] * (max_de_len - len(line)) for line in de_input_batch])
            de_target_batch = np.array([line + [0] * (max_de_len - len(line)) for line in de_target_batch])
            begin = end
            yield en_input_batch, de_input_batch, de_target_batch

    def one_epoch_generator_for_test(self, batch_size=config.test_batch_size):
        begin = 0
        while begin < self.test_size:
            end = begin + batch_size
            en_input_batch = self.test_data[0][begin:end]
            de_input_batch = self.test_data[1][begin:end]
            de_target_batch = self.test_data[2][begin:end]
            max_en_len = max([len(line) for line in en_input_batch])
            max_de_len = max([len(line) for line in de_input_batch])
            en_input_batch = np.array([line + [0] * (max_en_len - len(line)) for line in en_input_batch])
            de_input_batch = np.array([line + [0] * (max_de_len - len(line)) for line in de_input_batch])
            de_target_batch = np.array([line + [0] * (max_de_len - len(line)) for line in de_target_batch])
            begin = end
            yield en_input_batch, de_input_batch, de_target_batch


if __name__ == '__main__':
    d = Dataset()
    g = d.one_epoch_generator_for_test(1)
    print(next(g))