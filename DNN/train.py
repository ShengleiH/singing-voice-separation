import tensorflow as tf
from config import TrainConfig
from model import Model
from data_helper import get_mag_matrix, get_wavs, get_phase, to_wav, to_mag, to_stft, get_files
import numpy as np
from mir_eval.separation import bss_eval_sources
import os
import json

def train():
    model = Model(hidden_size=TrainConfig.HIDDEN_SIZE)

    #################################################
    #           session & summary writer            #
    #################################################
    sess = tf.Session(config=TrainConfig.session_conf)
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(TrainConfig.GRAPH_PATH, sess.graph)
    summary_op = summaries(model)

    #################################################
    #                     Load Data                 #
    #################################################
    mixed_mag_matrix, src1_mag_matrix, src2_mag_matrix, phase_matrix = get_mag_matrix()

    #################################################
    #                     Training                  #
    #################################################
    music_gnsdr = {}
    music_gsir = {}
    music_gsar = {}
    vocal_gnsdr = {}
    vocal_gsir = {}
    vocal_gsar = {}
        
    for step in range(TrainConfig.MAX_STEP):

        # Generate batches
        n_samples = mixed_mag_matrix.shape[-1]
        batch_indices = np.random.choice(n_samples, TrainConfig.BATCH_SIZE, replace=False)

        mixed_mag = mixed_mag_matrix[:, batch_indices]
        src1_mag = src1_mag_matrix[:, batch_indices]
        src2_mag = src2_mag_matrix[:, batch_indices]

        # reshape to feed_dict
        mixed_batch = mixed_mag.T
        src1_batch = src1_mag.T
        src2_batch = src2_mag.T

        feed_dict = {
            model.x_mixed: mixed_batch,
            model.y_src1: src1_batch,
            model.y_src2: src2_batch,
            model.keep_prob: 0.5
        }

        # start training
        loss, summary = model.train(sess, summary_op, feed_dict)
        print('Step {}, loss = {}'.format(step + 1, loss))

        # Write summary
        writer.add_summary(summary, global_step=step)

        #################################################
        #                   Save Model                  #
        #################################################
        if step % TrainConfig.CKPT_STEP == 0 or step + 1 == TrainConfig.MAX_STEP:
            # Save the updated model
            tf.train.Saver().save(sess, TrainConfig.CKPT_PATH, step)
        
        #################################################
        #               Evaluate Model                  #
        #################################################
        
        if step % TrainConfig.EVAL_STEP == 0 or step + 1 == TrainConfig.MAX_STEP:
            # Get validation set files
            dev_files = get_files(TrainConfig.DEV_PATH)

            gnsdr = np.zeros(2)
            gsir = np.zeros(2)
            gsar = np.zeros(2)
            total_len = 0

            for i, file in enumerate(dev_files):
                print('Validation test on file {}/{}'.format(i + 1, len(dev_files)))

                # Get each validation file
                dev_mixed_wav, dev_src1_wav, dev_src2_wav = get_wavs(file)

                dev_mixed_stft = to_stft(dev_mixed_wav)
                dev_mixed_mag = to_mag(dev_mixed_stft)
                dev_phase = get_phase(dev_mixed_stft)

                dev_mixed_batch = dev_mixed_mag.T

                feed_dict = {
                    model.x_mixed: dev_mixed_batch,
                    model.keep_prob: 1.0
                }

                # Predict vocal and music
                pred_dev_src1_batch, pred_dev_src2_batch = model.predict(sess, feed_dict)

                # Reshape to magnitude
                pred_dev_src1_mag = pred_dev_src1_batch.T
                pred_dev_src2_mag = pred_dev_src2_batch.T

                # Convert to wav
                pred_dev_src1_wav = to_wav(pred_dev_src1_mag, dev_phase)
                pred_dev_src2_wav = to_wav(pred_dev_src2_mag, dev_phase)

                wav_len = dev_mixed_wav.shape[-1]

                # BSS Evaluation
                sdr, sir, sar, _ = bss_eval_sources(np.array([dev_src1_wav, dev_src2_wav]),
                                                    np.array([pred_dev_src1_wav, pred_dev_src2_wav]), False)
                sdr_mixed, _, _, _ = bss_eval_sources(np.array([dev_src1_wav, dev_src2_wav]),
                                                      np.array([dev_mixed_wav, dev_mixed_wav]), False)

                nsdr = sdr - sdr_mixed
                gnsdr += wav_len * nsdr
                gsir += wav_len * sir
                gsar += wav_len * sar
                total_len += wav_len

            gnsdr = gnsdr / total_len
            gsir = gsir / total_len
            gsar = gsar / total_len

            # print('number of dev files = {}'.format(len(dev_files)))
            # print('music gnsdr = {}'.format(gnsdr[0]))
            # print('music gsir = {}'.format(gsir[0]))
            # print('music gsar = {}'.format(gsar[0]))
            # print('vocal gnsdr = {}'.format(gnsdr[1]))
            # print('vocal gsir = {}'.format(gsir[1]))
            # print('vocal gsar = {}'.format(gsar[1]))
            
            # Write score results to json file
            music_gnsdr[step] = gnsdr[0]
            music_gsir[step] = gsir[0]
            music_gsar[step] = gsar[0]
            vocal_gnsdr[step] = gnsdr[1]
            vocal_gsir[step] = gsir[1]
            vocal_gsar[step] = gsar[1]
        
    vocal_scores = {}
    music_scores = {}
    vocal_scores['gnsdr'] = vocal_gnsdr
    vocal_scores['gsir'] = vocal_gsir
    vocal_scores['gsar'] = vocal_gsar
    music_scores['gnsdr'] = music_gnsdr
    music_scores['gsir'] = music_gsir
    music_scores['gsar'] = music_gsar

    score_results = {}
    score_results['vocal_scores'] = vocal_scores
    score_results['music_scores'] = music_scores

    with open(TrainConfig.JSON_PATH, 'w') as result_file:
        json.dump(score_results, result_file)

def summaries(model):
    tf.summary.scalar('loss', model.loss)
    tf.summary.histogram('x_mixed', model.x_mixed)
    tf.summary.histogram('y_src1', model.y_src1)
    tf.summary.histogram('y_src2', model.y_src2)
    return tf.summary.merge_all()


def setup_path():
    if not os.path.exists(TrainConfig.CKPT_PATH):
        os.mkdir('checkpoints/')
        os.mkdir(TrainConfig.CKPT_PATH)

        
if __name__ == '__main__':
    setup_path()
    train()
