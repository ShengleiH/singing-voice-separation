import librosa
import numpy as np
from os import walk
from config import TrainConfig


def get_files(root_path):
    all_files = []
    path = ''

    for root, dirs, files in walk(root_path):
        path = root

    for file in files:
        new_file_path = path + '/' + file
        all_files.append(new_file_path)

    return sorted(all_files)


def pad_zeros(wav, hop_length=TrainConfig.HOP_LENGTH):
    wav_length = wav.shape[-1]
    new_wav = wav

    if wav.ndim > 1:
        temp = wav_length // hop_length
        if (wav_length != temp * hop_length):
            n_zeros = (temp + 1) * hop_length - wav_length
            new_wav = np.pad(new_wav, ((0, 0), (0, n_zeros)), 'constant', constant_values=0)

    else:
        temp = wav_length // hop_length
        if (wav_length != temp * hop_length):
            n_zeros = (temp + 1) * hop_length - wav_length
            new_wav = np.pad(new_wav, ((0, n_zeros)), 'constant', constant_values=0)

    return new_wav


def get_wavs(file):
    stereo = pad_zeros(librosa.load(file, TrainConfig.SR, mono=False)[0]) # (2, num_sample_points)
    src1 = stereo[0]
    src2 = stereo[1]
    mixed = librosa.to_mono(stereo)
    return mixed, src1, src2


def to_stft(wav, n_fft=TrainConfig.N_FFT, hop_length=TrainConfig.HOP_LENGTH):
    stft = librosa.stft(wav, n_fft, hop_length)
    return stft


def to_mag(stft):
    return np.abs(stft)


def get_phase(stft):
    return np.angle(stft)


def to_wav(mag, phase, hop_length=TrainConfig.HOP_LENGTH):
    stft = mag * np.exp(1.j * phase)
    wav = librosa.istft(stft, hop_length)
    return wav


def get_mag_matrix(root_path=TrainConfig.TRAIN_PATH):
    train_files = get_files(root_path)

    mixed_mag_matrix = []
    src1_mag_matrix = []
    src2_mag_matrix = []
    phase_matrix = []

    for i, file in enumerate(train_files):
        print('Process file {}/{}'.format(i + 1, len(train_files)))
        mixed_wav, src1_wav, src2_wav = get_wavs(file)

        # Get frames
        mixed_stft = to_stft(mixed_wav)
        src1_stft = to_stft(src1_wav)
        src2_stft = to_stft(src2_wav)

        # Get magnitude
        mixed_mag = to_mag(mixed_stft)
        src1_mag = to_mag(src1_stft)
        src2_mag = to_mag(src2_stft)

        phase = get_phase(mixed_stft)

        mixed_mag_matrix.append(mixed_mag)
        src1_mag_matrix.append(src1_mag)
        src2_mag_matrix.append(src2_mag)
        phase_matrix.append(phase)
        
        # Data augment if needed
        if TrainConfig.AUGMENTED:
            aug_mixed_mag_matrix, aug_src1_mag_matrix, aug_src2_mag_matrix, aug_phase_matrix = data_augmentation(file)
            mixed_mag_matrix.extend(aug_mixed_mag_matrix)
            src1_mag_matrix.extend(aug_src1_mag_matrix)
            src2_mag_matrix.extend(aug_src2_mag_matrix)
            phase_matrix.extend(aug_phase_matrix)

    mixed_mag_matrix = np.concatenate(mixed_mag_matrix, axis=1)  # (513, 43690)
    src1_mag_matrix = np.concatenate(src1_mag_matrix, axis=1)  # (513, 43690)
    src2_mag_matrix = np.concatenate(src2_mag_matrix, axis=1)  # (513, 43690)
    phase_matrix = np.concatenate(phase_matrix, axis=1)

    print('We totally have {} frames'.format(mixed_mag_matrix.shape[-1]))
    return mixed_mag_matrix, src1_mag_matrix, src2_mag_matrix, phase_matrix


def save_wav(data, path, sr=TrainConfig.SR):
    print('Saving data to {}'.format(path))
    librosa.output.write_wav(path, data, sr)
    
    
def data_augmentation(file, shift_step=TrainConfig.SHIFT_STEP):
    print('shift_step = {}'.format(shift_step))
    _, src1_wav, src2_wav = get_wavs(file)
    
    aug_mixed_mag_matrix = []
    aug_src1_mag_matrix = []
    aug_src2_mag_matrix = []
    aug_phase_matrix = []
    
    wav_length = src1_wav.shape[-1]
    rotate_step = min(shift_step, wav_length - 1)
    for i, point in enumerate(range(rotate_step, wav_length - rotate_step + 1, rotate_step)):
        # print('wav_length = {}'.format(wav_length))
        # print('rotate_step = {}'.format(rotate_step))
        print('rotate round {}/{}'.format(i+1, int(np.ceil((wav_length - 2 * rotate_step + 1) / rotate_step))))
        
        new_src2_wav = np.concatenate((src2_wav[point:], src2_wav[0: point]), axis=0)
        mixed_stereo = np.concatenate((src1_wav.reshape(1, wav_length), new_src2_wav.reshape(1, wav_length)), axis=0)
        new_mixed_wav = librosa.to_mono(mixed_stereo)
        
#         print('new_mixed_wav shape = {}'.format(new_mixed_wav.shape))
#         print('src1_wav shape = {}'.format(src1_wav.shape))
#         print('new_src2_wav shape = {}'.format(new_src2_wav.shape))
        
        # Get frames
        new_mixed_stft = to_stft(new_mixed_wav)
        src1_stft = to_stft(src1_wav)
        new_src2_stft = to_stft(new_src2_wav)

        # Get magnitude
        new_mixed_mag = to_mag(new_mixed_stft)
        src1_mag = to_mag(src1_stft)
        new_src2_mag = to_mag(new_src2_stft)

        new_phase = get_phase(new_mixed_stft)

        aug_mixed_mag_matrix.append(new_mixed_mag)
        aug_src1_mag_matrix.append(src1_mag)
        aug_src2_mag_matrix.append(new_src2_mag)
        aug_phase_matrix.append(new_phase)
    
    return aug_mixed_mag_matrix, aug_src1_mag_matrix, aug_src2_mag_matrix, aug_phase_matrix
    
    
# def get_train_batches(batch_size=TrainConfig.BATCH_SIZE, ):
#     mixed_mag_matrix, src1_mag_matrix, src2_mag_matrix, phase_matrix = get_mag_matrix()
#
#     # Generate batches
#     n_samples = mixed_mag_matrix.shape[-1]
#     batch_indices = np.random.choice(n_samples, batch_size, replace=False)
#
#     mixed_mag_batch = mixed_mag_matrix[:, batch_indices]
#     src1_mag_batch = src1_mag_matrix[:, batch_indices]
#     src2_mag_batch = src2_mag_matrix[:, batch_indices]
#     phase_batch = phase_matrix[:, batch_indices]
#
#     return mixed_mag_batch, src1_mag_batch, src2_mag_batch, phase_batch
