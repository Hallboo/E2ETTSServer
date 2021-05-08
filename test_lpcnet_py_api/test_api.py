
# Test LPCNET Python API
# f32 type feature to wave
# Zhang Haobo
# May 8, 2021

import random
import numpy as np
import lpcnet
import wave

if __name__=="__main__":

    # f32 file, (N_frame, 55) 32 bit float
    f32_path = "test_english_tacotron_lpcnet-00000-00271-feats.f32"
    wav_path = "test_english_tacotron_lpcnet-00000-00271-wave.wav"

    features = np.fromfile(f32_path, dtype='float32')
    features = np.resize(features, (-1, 55))

    print("Read f32 file: {} shape:{} sample should:{}".format(f32_path, features.shape, features.shape[0]*160))

    # run lpcnet for the segments
    pcm = lpcnet.run_for_chunk(features.tolist())
    pcm = np.array(pcm, dtype = "int16").reshape(-1)
    print("LPCNet Done {}, generate sample {}".format(pcm.shape, pcm.shape[0]))

    # save the wave file
    with wave.open(wav_path, 'wb') as fp:
        # 1 channel, 16 bits, 16000 sample rate
        fp.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        fp.writeframes(pcm)

    print("Save audio:{}".format(wav_path))

