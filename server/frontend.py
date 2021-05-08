
# simple frontend

import re
import logging

def Frontend(phone2id, idim, text):
    """Clean text and then convert to id sequence."""
    idseq = []
    T=re.split(r"",text)

    for w in T:
        if w == ' ':
            w = "<space>"

        if len(w) == 0:
            continue
        
        if w not in phone2id.keys():

            logging.info('%s => %s'%(w,w.lower()))
            w = w.lower()
            if not w in phone2id.keys():
                break
            idseq += [phone2id[w]]
        else:
            idseq += [phone2id[w]]

    idseq += [ idim - 1 ]  # <eos>

    return idseq

def LoadDictionary(dict_path):

    logging.info("Load Dictionary: {}".format(dict_path))

    with open(dict_path) as fp:
        lines = fp.readlines()
        lines = [ line.replace("\n", "").split(" ") for line in lines ]

    phone2id = { c: int(i) for c, i in lines }

    logging.info('Dictionary Size: {}'.format(len(phone2id)))

    return phone2id
