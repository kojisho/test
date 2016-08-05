import h5py
import theano
import theano.tensor as T
import numpy as np

if theano.config.device == 'cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from util import shared_zeros, shared_normal, shared_normal_vector
from attention import SelectiveAttentionModel
from collections import OrderedDict
import datetime
import sys
import json
import math
import cPickle as pickle
from random import randint

create_captions = __import__('create-captions')
create_mnist_captions_dataset = create_captions.create_mnist_captions_dataset

#assert(theano.config.scan.allow_gc == True), "set scan.allow_gc to True ; otherwise you will run out of gpu memory"
#assert(theano.config.allow_gc == True), "set allow_gc to True ; otherwise you will run out of gpu memory"

sys.stdout.flush()

floatX = theano.config.floatX

np.random.seed(np.random.randint(1 << 30))
rng = RandomStreams(seed=np.random.randint(1 << 30))

#np.random.seed(34)
#rng = RandomStreams(seed=34)

params_names = ['W_y_hLangEnc', 'W_hLangEnc_hLangEnc', 'b_hLangEnc', 'W_yRev_hLangEncRev', 'W_hLangEncRev_hLangEncRev', 'b_hLangEncRev', 'W_lang_align', 'W_hdec_align', 'b_align', 'v_align', 'W_s_hdec', 'W_hdec_read_attent', 'b_read_attent', 'W_henc_henc', 'W_inp_henc', 'b_henc', 'W_henc_mu', 'W_henc_logsigma', 'b_mu', 'b_logsigma', 'W_hdec_hdec', 'W_z_hdec', 'b_hdec', 'W_hdec_write_attent', 'b_write_attent', 'W_hdec_c', 'b_c', 'W_hdec_mu_and_logsigma_prior', 'b_mu_and_logsigma_prior', 'h0_lang', 'h0_enc', 'h0_dec', 'c0']



if __name__ == '__main__':
    model_name = "models/mnist-captions.json"
    with open(model_name) as model_file:
        model = json.load(model_file)
    dimY = int(model["model"][0]["dimY"])
    dimLangRNN = int(model["model"][0]["dimLangRNN"])
    dimAlign = int(model["model"][0]["dimAlign"])

    dimX = int(model["model"][0]["dimX"])
    dimReadAttent = int(model["model"][0]["dimReadAttent"])
    dimWriteAttent = int(model["model"][0]["dimWriteAttent"])
    dimRNNEnc = int(model["model"][0]["dimRNNEnc"])
    dimRNNDec = int(model["model"][0]["dimRNNDec"])
    dimZ = int(model["model"][0]["dimZ"])
    runSteps = int(model["model"][0]["runSteps"])



