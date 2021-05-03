
# ESPNET Tacotron LPCNET worker
# ESPNET and LPCNET ENV
# Zhang Haobo
# Mar 13, 2021

import os
import re
import sys

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

with open('config/config.yaml', 'r', encoding = 'utf-8') as fp:
    config = yaml.safe_load(fp)

device = torch.device(config['torch_device'])

def RunLPCNet(overlap, num_chunk_frame, vocoder_path, tmp_dir, sub_f32_path, req_id, sub_req_id, s1, s2):

    if s2 == s1 + overlap + num_chunk_frame + overlap:
        duration = num_chunk_frame
    else:
        duration = (s2 - s1 - overlap)

    if s1 != 0:
        start = overlap
    else:
        start = 0

    #print("s1:{} s2:{} start:{} Duration:{}".format(s1, s2, s1+start, s1+start+duration))

    start = start * 0.01
    duration = duration * 0.01

    sub_pcm_path = os.path.join(tmp_dir, req_id, sub_req_id + "-wave.pcm")
    sub_wav_path = os.path.join(tmp_dir, req_id, sub_req_id + "-wave.wav")
    os.system("{} -synthesis {} {}".format(vocoder_path, sub_f32_path, sub_pcm_path))

    os.system("sox -c 1 -r 16000 -t sw {} -t wav {} trim {} {}".format(
            sub_pcm_path, sub_wav_path, start, duration))

    return sub_wav_path

def CombineAudio(wav_batch_path, wav_path):
    wav_batch_path.sort()
    waves = " ".join(wav_batch_path)
    os.system("sox {} {}".format(waves, wav_path))

class TacotronLPCNetWorker():
    def __init__(self, config):

        self.tmp_dir = config['tmp']
        #os.makedirs(self.tmp_dir, exist_ok=True)
        self.debug_mode = config['debug_mode']

        self.phone2id = self.LoadDictionary(config['dict_path'])

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

        self.vocoder_path = config['vocoder_path']
        logging.info("Vocoder Path: {}".format(self.vocoder_path))

    def SaveAsF32(self, npy_feat, f32_path):

        features = np.zeros((npy_feat.shape[0], 55), dtype='float32')
        features[ : , :18 ]   = npy_feat[ :, :18 ]
        features[ : , 36:38 ] = npy_feat[ :, 18: ]

        with open(f32_path, 'wb') as fp:
            features.tofile(fp)
        logging.info("{} Save file: {}".format(__file__, f32_path))

    def SaveNPartAsF32(self, overlap, npy_feat, num_chunk_frame, req_dir, req_id):

        features = np.zeros((npy_feat.shape[0], 55), dtype='float32')
        features[ : , :18 ]   = npy_feat[ :, :18 ]
        features[ : , 36:38 ] = npy_feat[ :, 18: ]

        sub_range = int(features.shape[0]/num_chunk_frame) + 1
        #print(features.shape[0], sub_range)

        f32_batch_path = []
        for i in range(sub_range):
            
            s1 = i * num_chunk_frame

            s2 = i * num_chunk_frame + num_chunk_frame
            if s2 > features.shape[0]:
                s2 = features.shape[0]

            if s2 - s1 < overlap: break

            #print("A s1:{} s2:{}".format(s1, s2))

            if s2 < features.shape[0]:
                s2 = s2 + overlap
                if s2 > features.shape[0]:
                    s2 = features.shape[0]

            if i != 0: s1 = s1 - overlap

            #print("B s1:{} s2:{}".format(s1, s2))
 
            surfix = str(i).zfill(3)

            sub_req_id = "{}-{:05d}-{:05d}".format(req_id, s1,s2)
            f32_name = "{}-feats.f32".format(sub_req_id)
            f32_path = os.path.join(req_dir, f32_name)
            #print(f32_path)

            with open(f32_path, 'wb') as fp:
                part_features = features[s1:s2]
                part_features.tofile(fp)
                #logging.info("{} Save file: {}".format(__file__, f32_path))

                f32_batch_path.append([sub_req_id, f32_path, s1, s2])

        return f32_batch_path

    # def RunLPCNet(self, sub_f32_path, req_id, sub_req_id):

    #     sub_pcm_path = os.path.join(self.tmp_dir, req_id, sub_req_id + "-wave.pcm")
    #     sub_wav_path = os.path.join(self.tmp_dir, req_id, sub_req_id + "-wave.wav")

    #     os.system("{} {} {}".format(self.vocoder_path, sub_f32_path, sub_pcm_path))
    #     logging.info("{} Save file: {}".format(__file__, sub_pcm_path))
    #     os.system("sox -c 1 -r 16000 -t sw {} -t wav {}".format(sub_pcm_path, sub_wav_path))
    #     logging.info("{} Save file: {}".format(__file__, sub_wav_path))

    #     return sub_wav_path

    def Text2Speech(self, text, req_id):
        mat = self.Frontend(text)
        c, _, _ = self.model.inference(mat, self.inference_args)
        f=c.cpu().detach().numpy()

        logging.info("{} Pass Acoustic Model".format(__file__))

        req_dir = os.path.join(self.tmp_dir, req_id)

        os.makedirs(req_dir, exist_ok=True)

        f32_batch_path = self.SaveNPartAsF32(self.overlap, f, self.num_chunk_frame, req_dir, req_id)
        logging.info("{} Save F32 Done".format(__file__))

        executor = ProcessPoolExecutor(max_workers=len(f32_batch_path))
        futures = []

        for sub_req_id, sub_f32_path, s1, s2 in f32_batch_path:
            futures.append(executor.submit(
                partial(RunLPCNet, self.overlap, self.num_chunk_frame, self.vocoder_path,
                self.tmp_dir, sub_f32_path, req_id, sub_req_id, s1, s2)
            ))

        wav_batch_path = [ future.result() for future in futures ]

        wav_path = os.path.join(req_dir, req_id + "-wave.wav")
        CombineAudio(wav_batch_path, wav_path)

        logging.info("{} Wave {}".format(__file__, wav_path))

        if self.debug_mode:
            # numpy type feature
            np.save(os.path.join(req_dir, req_id + "-feats.npy"), f)
            # plot
            plt.figure()
            plt.matshow(np.flip(f.T))
            plt.savefig(os.path.join(req_dir, req_id + "-demo.png"), format="png")
            plt.close()

        return wav_path

    def Frontend(self, text):
        """Clean text and then convert to id sequence."""
        idseq = []
        T=re.split(r"",text)

        for w in T:
            if w == ' ':
                w = "<space>"

            if len(w) == 0:
                continue
            
            if w not in self.phone2id.keys():
                warnings.warn('%s => %s'%(w,w.lower()))
                w = w.lower()
                assert(w in phone2id.keys())
                idseq += [self.phone2id[w]]
            else:
                idseq += [self.phone2id[w]]

        idseq += [ self.idim - 1 ]  # <eos>

        return torch.LongTensor(idseq).view(-1).to(device)

    def LoadDictionary(self, dict_path):

        logging.info("Load Dictionary: {}".format(dict_path))

        with open(dict_path) as fp:
            lines = fp.readlines()
            lines = [ line.replace("\n", "").split(" ") for line in lines ]

        phone2id = { c: int(i) for c, i in lines }

        logging.info('Dictionary Size: {}'.format(len(phone2id)))

        return phone2id

if __name__ == "__main__":

    # code for test
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)8s %(asctime)s %(message)s ")
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
