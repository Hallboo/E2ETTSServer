
# ESPNET Tacotron LPCNET worker
# ESPNET and LPCNET ENV
# Zhang Haobo
# May 8, 2021

import os
import re
import sys
import wave
import time

import yaml
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

# define E2E-TTS model
from argparse import Namespace
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.utils.dynamic_import import dynamic_import

import lpcnet 
import frontend

with open('config/config.yaml', 'r', encoding = 'utf-8') as fp:
    config = yaml.safe_load(fp)

device = torch.device(config['torch_device'])

def run_vocoder(overlap, num_chunk_frame, tmp_dir, i, s1, s2, a1, a2, sub_feats):

    pcm = lpcnet.run_for_chunk(sub_feats.tolist())
    pcm = np.array(pcm, dtype = "int16")

    duration = a2 - a1

    if s1 != 0:
        start = overlap
    else:
        start = 0

    logging.debug("{} LPCNet API {:2d} {:4d} {:4d} {:4d} {:4d} shape:{} start {:2d} dur {:2d}".format(
        __file__, i, s1, s2, a1, a2, pcm.shape, start, duration))

    return (i, pcm[start:start+duration])

def Split(overlap, npy_feat, num_chunk_frame):

    features = np.zeros((npy_feat.shape[0], 55), dtype='float32')

    features[ : , :18 ]   = npy_feat[ :, :18 ]
    features[ : , 36:38 ] = npy_feat[ :, 18: ]

    sub_range = int(features.shape[0]/num_chunk_frame) + 1

    logging.debug("{} num_frame:{} sub_range:{}".format(
        __file__, features.shape[0], sub_range))

    # [ seg1(n1,55), seg2(n2,55), ... seg(N, 55) ]
    batch_feats = []
    for i in range(sub_range):
        
        a1 = i * num_chunk_frame

        a2 = i * num_chunk_frame + num_chunk_frame
        if a2 > features.shape[0]:
            a2 = features.shape[0]

        if a2 - a1 < overlap: break

        s1, s2 = a1, a2

        if s2 < features.shape[0]:
            s2 = s2 + overlap
            if s2 > features.shape[0]:
                s2 = features.shape[0]

        if i != 0: s1 = s1 - overlap

        logging.debug("{} Split [{}] {:4d} {:4d} {:4d} {:4d}".format(__file__, i, 
                        s1, s2, a1, a2))

        segment_feats = features[s1:s2]
        batch_feats.append([i, s1, s2, a1, a2, segment_feats])

    return batch_feats

class TacotronLPCNetWorker():
    def __init__(self, config):

        logging.info("Worker2 init start:")

        self.tmp_dir = config['tmp']
        #os.makedirs(self.tmp_dir, exist_ok=True)
        self.debug_mode = config['debug_mode']

        self.phone2id = frontend.LoadDictionary(config['dict_path'])

        self.idim, odim, train_args = get_model_conf(config['acoustic_model_path'])

        model_class = dynamic_import(train_args.model_module)
        model = model_class(self.idim, odim, train_args)
        torch_load(config['acoustic_model_path'], model)

        logging.info("Model Load Done {}".format(config['acoustic_model_path']))

        self.model = model.eval().to(device)
        self.inference_args = Namespace(**{
            "threshold": config['threshold'],
            "minlenratio": config['minlenratio'],
            "maxlenratio": config['maxlenratio']}
            )

        logging.info("TacotronLPCNetWorker Init Done")

        self.num_chunk_frame = config["num_chunk_frame"]
        logging.info("Chunk size: {}".format(self.num_chunk_frame))
        self.overlap = config["overlap"]       
        logging.info("Overlap: {}".format(self.overlap))

    def Text2Speech(self, text, req_id):

        time_start = time.time()

        # 1. front end
        idseq = frontend.Frontend(self.phone2id, self.idim, text)
        mat = torch.LongTensor(idseq).view(-1).to(device)
        time_frontend = time.time()

        # 2. acoustic model inference
        c, _, _ = self.model.inference(mat, self.inference_args)
        features = c.cpu().detach().numpy()

        logging.info("{} Pass Acoustic Model".format(__file__))
        time_acoustic = time.time()

        req_dir = os.path.join(self.tmp_dir, req_id)

        os.makedirs(req_dir, exist_ok=True)

        # 3. vocoder
        batch_feats = Split(self.overlap, features, self.num_chunk_frame)
        logging.info("{} Split {} done".format(__file__, len(batch_feats)))

        executor = ProcessPoolExecutor(max_workers=len(batch_feats))
        futures = []

        num_frame = 0
        for i, s1, s2, a1, a2, sub_feats in batch_feats:
            num_frame += a2 - a1
            futures.append(executor.submit(
                partial(run_vocoder, self.overlap, self.num_chunk_frame, self.tmp_dir,
                        i, s1, s2, a1, a2, sub_feats)
            ))

        batch_pcm = [ future.result() for future in futures ]

        pcm = np.zeros((num_frame, 160), dtype = "int16")

        index = 0
        for one_pcm in batch_pcm:
            i, sub_pcm = one_pcm
            #print(i, sub_pcm.shape)            
            start = index 
            num   = sub_pcm.shape[0]
            #print(start, num)
            pcm[start: start + num ]  = sub_pcm
            index += num

        pcm = np.reshape(pcm, -1 )[: num_frame * 160 ]

        wav_path = os.path.join(req_dir, req_id + "-wave.wav")

        with wave.open(wav_path, 'wb') as fp:
            # 1 channel, 16 bits, 16000 sample rate
            fp.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
            fp.writeframes(pcm)

        audio_dur = num_frame * 0.01
        logging.info("{} Wave {:0.3f}s {}".format(__file__, audio_dur, wav_path))

        if self.debug_mode:
            # numpy type feature
            np.save(os.path.join(req_dir, req_id + "-feats.npy"), features)
            # plot
            plt.figure()
            plt.matshow(np.flip(features.T))
            plt.savefig(os.path.join(req_dir, req_id + "-demo.png"), format="png")
            plt.close()

        time_vocoder = time.time()

        time_count_frontend = time_frontend - time_start
        time_count_acoustic = time_acoustic - time_frontend
        time_count_vocoder  = time_vocoder  - time_acoustic

        logging.info("{} Acoustic:{:0.3f}s Vocoder:{:0.3f}s A+V: {:0.3f}s".format(
            __file__, time_count_acoustic, time_count_vocoder, time_vocoder - time_start))

        return wav_path


if __name__ == "__main__":

    # code for test
    # log_level = logg.DEBUG
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)8s %(asctime)s %(message)s ")
    logging.info('Starting Up Tacotron LPCNet Worker')

    with open('config/config.yaml', 'r', encoding = 'utf-8') as fp:
        config = yaml.safe_load(fp)

    for k,v in config.items():
        logging.info("Config {}: {}".format(k,v))

    worker = TacotronLPCNetWorker(config)
    #worker.Text2Speech(
    #    'SIL ni1 hao3 wo1 shi1 xiao3 ai4 tong2 xue2 SIL',
    #    'test-test-test')

    logging.info("we would hope to make progress on that next year")
    worker.Text2Speech(
        'we would hope to make progress on that next year',
        'test_english_tacotron_lpcnet')
