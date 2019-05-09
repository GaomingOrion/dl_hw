import tensorflow as tf
import numpy as np
from model import Model
from dataset import Dataset
from common import config

from nltk.translate import bleu_score

def eval(model_path):
    # prepare dataset
    dataset = Dataset()
    config.input_vocab_size = dataset.encoder_vocab_size
    config.label_vocab_size = dataset.decoder_vocab_size
    config.train_size = dataset.train_size
    config.test_size = dataset.test_size

    # define computing graph
    model = Model()

    # session
    saver = tf.train.Saver()
    sess = tf.Session()

    # translate one sentence
    def predict_one_sentence(x, sess):
        de_inp = [[dataset.decoder_vocab[0]['<GO>']]]
        while True:
            y = np.array(de_inp)
            preds = sess.run(model.preds, {model.x: x, model.de_inp: y, model.is_training: False})
            if preds[0][-1] == dataset.decoder_vocab[0]['<EOS>']:
                break
            de_inp[0].append(preds[0][-1])
        res = [dataset.decoder_vocab[1][idx] for idx in de_inp[0][1:]]
        return res

    # record result
    true_sentences, pred_sentences = [], []

    with sess.as_default():
        print('Restore model from {:s}'.format(model_path))
        saver.restore(sess=sess, save_path=model_path)
        for en_input_batch, _, de_target_batch in dataset.one_epoch_generator_for_test(batch_size=1):
            pred_sentences.append(predict_one_sentence(en_input_batch, sess))
            true_sentences.append([dataset.decoder_vocab[1][idx] for idx in de_target_batch[0][:-1]])

    score = np.mean([bleu_score.sentence_bleu([true_sentences[i]], pred_sentences[i]) for i in range(len(true_sentences))])
    print(score)

    # save results
    def save_sentences(sentences, path):
        with open(path, 'w', encoding='utf-8') as f:
            for s in sentences:
                f.write(' '.join(s) + '\n')
    save_sentences(true_sentences, '.\\result\\true.txt')
    save_sentences(pred_sentences, '.\\result\\pred.txt')

    return true_sentences, pred_sentences

if __name__ == '__main__':
    a, b = eval('.\\tf_ckpt\\model-adam-e41-loss_4.0455.ckpt-41')