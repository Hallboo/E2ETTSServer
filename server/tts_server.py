#!/usr/bin/python3
#encoding:utf-8

# ENV: python3
# Zhang Haobo
# Mar 13, 2020
# speech synthesis demo server
# Acoustic Model: Tacotron2 trained by ESPNet
# Vocoder Model: LPCNet

import sys
from random import randint
import logging
import json
import os.path
import uuid
import time
import threading
import wave
import yaml

import asyncio
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.concurrent
# import concurrent.futures
# import common
# import tornado.options
#import tornado.gen

# from worker import TacotronLPCNetWorker

class WorkerLoop(threading.Thread):
    def __init__(self, application):
        threading.Thread.__init__(self)
        self.app = application
        self.worker = application.worker

    # worker main loop take text to generate audio
    def run(self):
        while self.app.run_worker_loop:
            # if jobs list have a request.
            if len(self.app.jobs) != 0:
                one_request = self.app.jobs.pop()
                logging.info("%s: pop text start generate : %s" % (one_request.name, one_request.text))
                # TODO if socket closed, skip the request
                #sample_r, wave = self.worker.tts(one_request.text, one_request.name)
                try:
                    wav_path = self.worker.Text2Speech(one_request.text, one_request.name)
                    logging.info("%s: generate audio: %s" % (one_request.name, wav_path))
                    one_request.socket.send_audio(one_request, wav_path)

                except KeyboardInterrupt:
                    sys.exit(1)

                except Exception as e:
                    logging.info(e)
                    one_request.socket.close()

            else:
                time.sleep(0.1)
                # print('Worker Play.')

class Application(tornado.web.Application):
    def __init__(self, config):
        settings = dict(
            cookie_secret="43oETzKXQAGaYdkL5gEmGeJJFuYh7EQnp2XdTP1o/Vo=",
            template_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
            static_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static"),
            xsrf_cookies=False,
            autoescape=None,
        )

        handlers = [(config['url'], ClientSocketHandler)]
        tornado.web.Application.__init__(self, handlers, **settings)

        # tts component
        if config['worker'] == "worker":
            from worker import TacotronLPCNetWorker
        else:
            from worker2 import TacotronLPCNetWorker

        self.worker = TacotronLPCNetWorker(config)
        
        # jobs sequense, all client instance
        self.jobs = []
        
        # current state request number
        self.request_num = 0

        self.wl = WorkerLoop(self)
        self.run_worker_loop = True
        self.wl.start()
        

class Request:
    def __init__(self, name, text, socket):
        self.name = name
        self.text = text
        self.socket = socket

class ClientSocketHandler(tornado.websocket.WebSocketHandler):
    # needed for Tornado 4.0
    def check_origin(self, origin):
        return True

    def send_audio(self, req, wav_path):
        logging.info("%s: Sending audio %s" % (self.id, wav_path))

        wave_file_point = wave.open(wav_path, 'rb')

        channels, width, framerate, frames = wave_file_point.getparams()[:4]

        param = dict(name=self.id, req_id = req.name, channels=channels, width=width, 
                    framerate=framerate, frames=frames)

        wav = wave_file_point.readframes(frames)

        self.write_message(json.dumps(param))
        self.write_message(wav, binary=True)

        message = dict(AUDIO_END=True)
        self.write_message(message, binary = False)
    
        wave_file_point.close()

    def open(self):
        self.id = str(uuid.uuid4())
        self.application.request_num += 1
        logging.info("%s: open(), current request numbers: %d" % (self.id,self.application.request_num))

    def on_connection_close(self):
        logging.info("%s: Handling on_connection_close()" % self.id)
        self.application.request_num -=1

    def on_message(self, message_str):
        message = json.loads(str(message_str))
        if 'text' in message:
            text = message['text']
            t = time.localtime(time.time())
            name = '%02s-%02d-%02d-%02d-%02d-%02d-%02d' %(str(t.tm_year)[2:], t.tm_mon, t.tm_mday,
                                                        t.tm_hour, t.tm_min, t.tm_sec, randint(0,99))
            req = Request(name, text, self)
            logging.info("Receive text : %s from %s" % (req.name, self.id))
            self.application.jobs.append(req)

def main():    
    logging.basicConfig(level=logging.INFO, format="%(levelname)8s %(asctime)s %(message)s ")
    logging.debug('Starting Up TTS Server')
    logging.debug('Based on ESPNET and LPCNet')

    with open('config/config.yaml', 'r', encoding = 'utf-8') as fp:
        config = yaml.safe_load(fp)

    for k,v in config.items():
        logging.debug("Config {}: {}".format(k,v))

    asyncio.set_event_loop(asyncio.new_event_loop())

    app = Application(config)
    
    server_port = config['port']
    app.listen(server_port)
    logging.info("[ START ] Server port : %05d" % server_port)
    tornado.ioloop.IOLoop.instance().start()
    app.run_worker_loop = False
    
if __name__ == "__main__":
    main()
