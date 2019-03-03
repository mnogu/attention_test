import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

import model

def evaluate(sentence, encoder, decoder, inp_lang, targ_lang,
             max_length_inp, max_length_targ):
    sentence = model.preprocess_sentence(sentence)

    inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = sequence.pad_sequences([inputs], maxlen=max_length_inp,
                                    padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, model.UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)

    for _ in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weigths to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.idx2word[predicted_id] + ' '

        if targ_lang.idx2word[predicted_id] == '<end>':
            return result, sentence

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence

def translate(sentence, encoder, decoder, inp_lang, targ_lang,
              max_length_inp, max_length_targ):
    result, sentence = evaluate(sentence, encoder, decoder,
                                inp_lang, targ_lang,
                                max_length_inp,
                                max_length_targ)

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))


def main():
    tf.enable_eager_execution()

    _, _, inp_lang, targ_lang, \
        max_length_inp, max_length_targ = model.load_dataset()

    optimizer = model.create_optimizer()

    encoder = model.create_encoder(inp_lang)
    decoder = model.create_decoder(targ_lang)

    checkpoint_dir, checkpoint = model.create_checkpoint(optimizer,
                                                         encoder, decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    translate('hace mucho frio aqui.', encoder, decoder, inp_lang, targ_lang,
              max_length_inp, max_length_targ)

if __name__ == '__main__':
    main()
