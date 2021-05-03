import argparse
import ws4py
from ws4py.client.threadedclient import WebSocketClient
import time
import threading
import sys
import json
import time
import wave
import os

class MyClient(WebSocketClient):

    def __init__(self, url, protocols=None, extensions=None, heartbeat_freq=None,
                 save_adaptation_state_filename=None, send_adaptation_state_filename=None):
        super(MyClient, self).__init__(url, protocols, extensions, heartbeat_freq)
        self.syn_dir = 'csyn'
        self.wave = None
        self.wave_head = None

    def opened(self):
        print("## LOG Socket opened!")
        message = dict(text = 'we would hope to make progress on that next year')
        self.send(json.dumps(message))
        print("## LOG Send message %s"%message)

    def received_message(self, message):
        if isinstance(message, ws4py.messaging.TextMessage):
            message = json.loads(str(message))
            if "name" in message:
                self.wave_head = message

            elif message["AUDIO_END"]:

                # {'name': '423b0e49-2fe2-47d9-84c3-9fad0577541b', 'req_id': '21-03-21-17-01-24-99', 'channels': 1, 'width': 2, 'framerate': 16000, 'frames': 42720}
                wav_path = self.wave_head["req_id"] + '.wav'
                with wave.open(wav_path, 'wb') as fp:
                    fp.setnchannels(self.wave_head['channels'])
                    fp.setsampwidth(self.wave_head['width'])
                    fp.setframerate(self.wave_head['framerate'])
                    fp.writeframes(self.wave)

                    self.wave = None
                    self.wave_head = None

                    print("## LOG Saved {}".format(wav_path))

            self.close()

        elif isinstance(message, ws4py.messaging.BinaryMessage):
            self.wave = message.data

    def closed(self, code, reason=None):
        print('## LOG Connection closed.')


def main():
    url = 'ws://localhost:10000/client/ws/tts'
    try:
        ws = MyClient(url)
        ws.connect()
        ws.run_forever()
    except KeyboardInterrupt:
        ws.close()

if __name__ == "__main__":
    main()

