import os
import shutil
import numpy as np
import tensorflow as tf

from model import Model
from config import EvalConfig
from data_helper import get_files, get_wavs, to_stft, to_mag, get_phase, to_wav, save_wav
from mir_eval.separation import bss_eval_sources

def eval():
    model = Model(hidden_size=EvalConfig.HIDDEN_SIZE)
    
    sess = tf.Session(config=EvalConfig.session_conf)
    sess.run(tf.global_variables_initializer())
    
    ckpt = tf.train.get_checkpoint_state(EvalConfig.CKPT_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
    
    all_files = get_files(EvalConfig.EVAL_PATH)
    
    gnsdr = np.zeros(2)
    gsir = np.zeros(2)
    gsar = np.zeros(2)
    total_len = 0
        
    for i, wav_file in enumerate(all_files):
        print('Eval on file {}/{}, {}'.format(i+1, len(all_files), wav_file))
        mixed_wav, src1_wav, src2_wav = get_wavs(wav_file)
        
        # prepare data
        mixed_spec = to_stft(mixed_wav)
        mixed_mag = to_mag(mixed_spec)
        mixed_phase = get_phase(mixed_spec)
        
        src1_spec = to_stft(src1_wav)
        src1_phase = get_phase(src1_spec)
        
        src2_spec = to_stft(src2_wav)
        src2_phase = get_phase(src2_spec)
        
        mixed_batch = mixed_mag.T
        
        # separate wav
        feed_dict = {
            model.x_mixed: mixed_batch,
            model.keep_prob: 1.0
        }
        
        pred_src1_batch, pred_src2_batch = model.predict(sess, feed_dict)
        
        # transfer to wav
        pred_src1_mag = pred_src1_batch.T
        pred_src2_mag = pred_src2_batch.T
                
        pred_src1_wav = to_wav(pred_src1_mag, mixed_phase)
        pred_src2_wav = to_wav(pred_src2_mag, mixed_phase)
        
        if EvalConfig.SAVE_FILE:
            wav_file_name = wav_file.split('/')[-1].split('.')[0]
            original_path = os.path.join(EvalConfig.RESULT_PATH, wav_file_name + '_original.wav')
            music_path = os.path.join(EvalConfig.RESULT_PATH, wav_file_name + '_music.wav')
            vocal_path = os.path.join(EvalConfig.RESULT_PATH, wav_file_name + '_vocal.wav')
            save_wav(mixed_wav, original_path)
            save_wav(pred_src1_wav, music_path)
            save_wav(pred_src2_wav, vocal_path)
        
        # BSS_EVAL
        wav_len = mixed_wav.shape[-1]
        sdr, sir, sar, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                                    np.array([pred_src1_wav, pred_src2_wav]), False)
        sdr_mixed, _, _, _ = bss_eval_sources(np.array([src1_wav, src2_wav]),
                                              np.array([mixed_wav, mixed_wav]), False)
        
        nsdr = sdr - sdr_mixed
        gnsdr += wav_len * nsdr
        gsir += wav_len * sir
        gsar += wav_len * sar
        total_len += wav_len

    gnsdr = gnsdr / total_len
    gsir = gsir / total_len
    gsar = gsar / total_len
    # Write the score of BSS metrics
    print('GNSDR_music={}'.format(gnsdr[0]))
    print('GSIR_music={}'.format(gsir[0]))
    print('GSAR_music={}'.format(gsar[0]))
    print('GNSDR_vocal={}'.format(gnsdr[1]))
    print('GSIR_vocal={}'.format(gsir[1]))
    print('GSAR_vocal={}'.format(gsar[1]))
            

def setup_path():
    if not os.path.exists(EvalConfig.RESULT_PATH):
        os.makedirs(EvalConfig.RESULT_PATH)

    
if __name__ == '__main__':
    setup_path()
    eval()

