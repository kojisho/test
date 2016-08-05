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

from func0805 import *

create_captions = __import__('create-captions')
create_mnist_captions_dataset = create_captions.create_mnist_captions_dataset

sys.stdout.flush()

class ReccurentAttentionVAE():

    def __init__(self, dimY, dimLangRNN, dimAlign, dimX, dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ, runSteps, batch_size, reduceLRAfter, inputData, inputLabels, valData=None, valLabels=None, pathToWeights=None):
        self.dimY = dimY
        self.dimLangRNN = dimLangRNN
        self.dimAlign = dimAlign
        self.dimX = dimX
        self.dimReadAttent = dimReadAttent
        self.dimWriteAttent = dimWriteAttent
        self.dimRNNEnc = dimRNNEnc
        self.dimRNNDec = dimRNNDec
        self.dimZ = dimZ
        self.runSteps = runSteps
        self.batch_size = batch_size
        self.reduceLRAfter = reduceLRAfter
        self.pathToWeights = pathToWeights

        self.inputData = inputData
        self.inputLabels = inputLabels  

        self.banned = [randint(0,10) for i in xrange(12)]

        print 'Banned configurations are :'
        print self.banned

        inputImages, inputCaptions, inputCounts = create_mnist_captions_dataset(inputData, inputLabels, self.banned)
        print 'Train Dataset'
        print inputImages.shape, inputCaptions.shape, inputCounts

        self.train_data = theano.shared(inputImages)
        self.train_captions = theano.shared(inputCaptions)
        self.input_shape = inputImages.shape

        del inputImages
        del inputCaptions

        if valData != None:
            valImages, valCaptions, valCounts = create_mnist_captions_dataset(valData, valLabels, self.banned)
            print 'Val Dataset'
            print valImages.shape, valCaptions.shape, valCounts

            self.val_data = theano.shared(valImages)
            self.val_captions = theano.shared(valCaptions)
            self.val_shape = valData.shape
            del valImages
            del valCaptions

        self._kl_final, self._logpxz, self._log_likelihood, self._c_ts, self._c_ts_gener, self._x, self._y, self._run_steps, self._updates_train, self._updates_gener, self._read_attent_params, self._write_attent_params, self._write_attent_params_gener, self._alphas_gener, self._params, self._mu_prior_t_gener, self._log_sigma_prior_t_gener = build_lang_encoder_and_attention_vae_decoder(self.dimY, self.dimLangRNN, self.dimAlign, self.dimX, self.dimReadAttent, self.dimWriteAttent, self.dimRNNEnc, self.dimRNNDec, self.dimZ, self.runSteps, self.pathToWeights)

    def _build_train_function(self):
        print 'building gradient function'
        t1 = datetime.datetime.now()
        gradients = T.grad(self._log_likelihood, self._params)
        t2 = datetime.datetime.now()
        print(t2-t1)

        self._index_cap = T.vector(dtype='int32') # index to the minibatch
        self._index_im = T.vector(dtype='int32')
        self._lr = T.scalar('lr', dtype=floatX)

        # Currently use AdaGrad & threshold gradients
        his = []
        for param in self._params:
            param_value_zeros = param.get_value() * 0
            his.append(theano.shared(param_value_zeros))

        threshold = 10.0
        decay_rate = 0.9

        self._updates_train_and_params = OrderedDict()
        self._updates_train_and_params.update(self._updates_train)
        
        for param, param_his, grad in zip(self._params, his, gradients):
            l2_norm_grad = T.sqrt(T.sqr(grad).sum())
            multiplier = T.switch(l2_norm_grad < threshold, 1, threshold / l2_norm_grad)
            grad = multiplier * grad

            param_his_new = decay_rate * param_his + (1 - decay_rate) * grad**2

            self._updates_train_and_params[param_his] = param_his_new
            self._updates_train_and_params[param] = param - (self._lr / T.sqrt(param_his_new + 1e-6)) * grad

        print 'building train function'
        t1 = datetime.datetime.now()
        self._train_function = theano.function(inputs=[self._index_im, self._index_cap, self._lr, self._run_steps], 
                                                outputs=[self._kl_final, self._logpxz, self._log_likelihood, self._c_ts, self._read_attent_params, self._write_attent_params],
                                                updates=self._updates_train_and_params,
                                                givens={
                                                    self._x: self.train_data[self._index_im],
                                                    self._y: self.train_captions[self._index_cap]
                                                })
        t2 = datetime.datetime.now()
        print (t2-t1)

    def train(self, lr, epochs, save=False, savedir=None, validateAfter=0):
        self._build_train_function()
        sys.stdout.flush()

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

    global height
    global width

    height = int(math.sqrt(dimX))
    width = int(math.sqrt(dimX))

    validateAfter = int(model["validateAfter"])
    save = bool(model["save"])
    lr = float(model["lr"])
    epochs = int(model["epochs"])
    batch_size = int(model["batch_size"])
    reduceLRAfter = int(model["reduceLRAfter"])
    pathToWeights = str(model["pathToWeights"])
    if pathToWeights == "None":
        pathToWeights = None

    dimension = int(math.sqrt(dimX / 3))
    if "data" in model:
        train_data_key = model["data"]["train_data"]["key"]
        train_data_file = model["data"]["train_data"]["file"]
        train_labels_key = model["data"]["train_labels"]["key"]
        train_labels_file = model["data"]["train_labels"]["file"]

        val_data_key = model["data"]["validation_data"]["key"]
        val_data_file = model["data"]["validation_data"]["file"]
        val_labels_key = model["data"]["validation_labels"]["key"]
        val_labels_file = model["data"]["validation_labels"]["file"]
    else:
        train_file = "/ais/gobi3/u/nitish/mnist/mnist.h5"
        train_key = "train"
        val_file = "/ais/gobi3/u/nitish/mnist/mnist.h5"
        val_key = "validation"

    train_data = np.copy(h5py.File(train_data_file, 'r')[train_data_key])
    train_labels = np.copy(h5py.File(train_labels_file, 'r')[train_labels_key])

    val_data = np.copy(h5py.File(val_data_file, 'r')[val_data_key])
    val_labels = np.copy(h5py.File(val_labels_file, 'r')[val_labels_key])
    print train_data.shape, train_labels.shape, val_data.shape, val_labels.shape

    savedir = None
    if "savedir" in model:
        savedir = model["savedir"]

    rvae = ReccurentAttentionVAE(dimY, dimLangRNN, dimAlign, dimX, dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec,
                                 dimZ, runSteps, batch_size, reduceLRAfter, train_data, train_labels, valData=val_data,
                                 valLabels=val_labels, pathToWeights=pathToWeights)
    rvae.train(lr, epochs, save=save, savedir=savedir, validateAfter=validateAfter)
