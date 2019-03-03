import os
import time

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import model

def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real,
                                                           logits=pred) * mask
    return tf.reduce_mean(loss_)

def main():
    tf.enable_eager_execution()

    input_tensor, target_tensor, inp_lang, targ_lang, \
        _, _ = model.load_dataset()

    # Creating training and validation sets using an 80-20 split
    input_tensor_train, _, target_tensor_train, _ \
        = train_test_split(input_tensor, target_tensor, test_size=0.2)

    buffer_size = len(input_tensor_train)
    n_batch = buffer_size // model.BATCH_SIZE

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train,
                                                  target_tensor_train))\
                                                     .shuffle(buffer_size)
    dataset = dataset.batch(model.BATCH_SIZE, drop_remainder=True)

    encoder = model.create_encoder(inp_lang)
    decoder = model.create_decoder(targ_lang)

    optimizer = model.create_optimizer()

    checkpoint_dir, checkpoint = model.create_checkpoint(optimizer,
                                                         encoder, decoder)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    epochs = 10

    for epoch in range(epochs):
        start = time.time()

        hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0

            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)
                dec_hidden = enc_hidden

                dec_input = tf.expand_dims([targ_lang.word2idx['<start>']]
                                           * model.BATCH_SIZE, 1)

                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_output)

                    loss += loss_function(targ[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))
            total_loss += batch_loss

            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            if batch % 100 == 0:
                print('Epoch {} Batch {} '
                      'Loss {:.4f}'.format(epoch + 1,
                                           batch,
                                           batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / n_batch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

if __name__ == '__main__':
    main()
