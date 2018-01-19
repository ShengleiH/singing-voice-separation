import tensorflow as tf


class ModelConfig:
    LR = 0.01
    N_FFT = 1024
    HOP_LENGTH = N_FFT // 2
    HIDDEN_SIZE = 1024
    LOSS = 'L1'
    
class DataConfig:
    BATCH_SIZE = 758
    # MAX_STEP = 186000
    # MAX_STEP = 300000
    # MAX_STEP = 30
    MAX_STEP = 10000

    AUGMENTED = True
    SHIFT_STEP = 36500  # approx 141000 frames
    # SHIFT_STEP = 10000
    
    
class TrainConfig:
    TRAIN_PATH = '../../data/Wavfile_divided_new/train'
    DEV_PATH = '../../data/dev'
    
    AUGMENTED = DataConfig.AUGMENTED
    SHIFT_STEP = DataConfig.SHIFT_STEP

    MAX_STEP = DataConfig.MAX_STEP
    CKPT_STEP = 1000
    BATCH_SIZE = DataConfig.BATCH_SIZE
    HIDDEN_SIZE = ModelConfig.HIDDEN_SIZE
    # EVAL_STEP = 10000
    EVAL_STEP = 1000

    SR = 16000
    N_FFT = 1024
    HOP_LENGTH = N_FFT // 2
    
    CASE = '{}_{}hiddenSize_{}batchSize_{}steps_{}lr_{}Augmented_{}shiftedSteps_mir1k'.format(ModelConfig.LOSS, ModelConfig.HIDDEN_SIZE, DataConfig.BATCH_SIZE, DataConfig.MAX_STEP, ModelConfig.LR, DataConfig.AUGMENTED, DataConfig.SHIFT_STEP)
    GRAPH_PATH = 'graphs/' + CASE + '/'
    CKPT_PATH = 'checkpoints/' + CASE + '/'
    JSON_PATH = 'json_results/' + CASE + '.json'
    
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.25
        ),
        intra_op_parallelism_threads=8
    )
    

class EvalConfig:
    # EVAL_PATH = '../../data/dev'
    EVAL_PATH = '../../data/Wavfile_divided_new/test'
    HIDDEN_SIZE = ModelConfig.HIDDEN_SIZE
    CASE = '{}_{}hiddenSize_{}batchSize_{}steps_{}lr_{}Augmented_{}shiftedSteps_mir1k'.format(ModelConfig.LOSS, ModelConfig.HIDDEN_SIZE, DataConfig.BATCH_SIZE, DataConfig.MAX_STEP, ModelConfig.LR, DataConfig.AUGMENTED, DataConfig.SHIFT_STEP)
    GRAPH_PATH = 'graphs/' + CASE + '/'
    CKPT_PATH = 'checkpoints/' + CASE + '/'
    SAVE_FILE = True
    
    RESULT_PATH = 'results/' + CASE + '_mixedPhase'
    
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.25
        ),
        intra_op_parallelism_threads=8
    )
