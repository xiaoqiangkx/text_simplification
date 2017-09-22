import tensorflow as tf

def get_session_config(model_config):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = model_config.allow_growth
    config.gpu_options.per_process_gpu_memory_fraction = model_config.per_process_gpu_memory_fraction
    if not model_config.use_gpu:
        config = tf.ConfigProto(
            device_count={'CPU': 0}
        )
        print('Not use GPU.')
    else:
        print('Use GPU.')
    return config