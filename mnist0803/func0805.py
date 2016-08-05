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

floatX = theano.config.floatX
np.random.seed(np.random.randint(1 << 30))
rng = RandomStreams(seed=np.random.randint(1 << 30))

params_names = ['W_y_hLangEnc', 'W_hLangEnc_hLangEnc', 'b_hLangEnc', 'W_yRev_hLangEncRev', 'W_hLangEncRev_hLangEncRev', 'b_hLangEncRev', 'W_lang_align', 'W_hdec_align', 'b_align', 'v_align', 'W_s_hdec', 'W_hdec_read_attent', 'b_read_attent', 'W_henc_henc', 'W_inp_henc', 'b_henc', 'W_henc_mu', 'W_henc_logsigma', 'b_mu', 'b_logsigma', 'W_hdec_hdec', 'W_z_hdec', 'b_hdec', 'W_hdec_write_attent', 'b_write_attent', 'W_hdec_c', 'b_c', 'W_hdec_mu_and_logsigma_prior', 'b_mu_and_logsigma_prior', 'h0_lang', 'h0_enc', 'h0_dec', 'c0']

global height
global width

height = int(math.sqrt(3600))
width = int(math.sqrt(3600))

def create_lstm_weights(dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ):
    W_hdec_read_attent = shared_normal(dimRNNDec, 5)
    b_read_attent = shared_zeros(5)

    W_henc_henc = shared_normal(dimRNNEnc, 4 * dimRNNEnc)
    W_inp_henc = shared_normal(dimReadAttent*dimReadAttent + dimReadAttent*dimReadAttent + dimRNNDec, 4 * dimRNNEnc)
    b_henc = shared_zeros(4 * dimRNNEnc)

    W_henc_mu = shared_normal(dimRNNEnc, dimZ)
    W_henc_logsigma = shared_normal(dimRNNEnc, dimZ)
    b_mu = shared_zeros(dimZ)
    b_logsigma = shared_zeros(dimZ)

    W_hdec_hdec = shared_normal(dimRNNDec, 4 * dimRNNDec)
    W_z_hdec = shared_normal(dimZ, 4 * dimRNNDec)
    b_hdec = shared_zeros(4 * dimRNNDec)

    W_hdec_write_attent = shared_normal(dimRNNDec, 5)
    b_write_attent = shared_zeros(5)

    W_hdec_c = shared_normal(dimRNNDec, dimWriteAttent*dimWriteAttent)
    b_c = shared_zeros(dimWriteAttent*dimWriteAttent)

    return W_hdec_read_attent, b_read_attent, W_henc_henc, W_inp_henc, b_henc, W_henc_mu, W_henc_logsigma, b_mu, b_logsigma, W_hdec_hdec, W_z_hdec, b_hdec, W_hdec_write_attent, b_write_attent, W_hdec_c, b_c


def create_lang_encoder_weights(dimY, dimLangRNN):
    W_y_hLangEnc = shared_normal(dimY, 4 * dimLangRNN)
    W_hLangEnc_hLangEnc = shared_normal(dimLangRNN, 4 * dimLangRNN)
    b_hLangEnc = shared_zeros(4 * dimLangRNN)

    W_yRev_hLangEncRev = shared_normal(dimY, 4 * dimLangRNN)
    W_hLangEncRev_hLangEncRev = shared_normal(dimLangRNN, 4 * dimLangRNN)
    b_hLangEncRev = shared_zeros(4 * dimLangRNN)

    return W_y_hLangEnc, W_hLangEnc_hLangEnc, b_hLangEnc, W_yRev_hLangEncRev, W_hLangEncRev_hLangEncRev, b_hLangEncRev

def create_align_weights(dimLangRNN, dimAlign, dimRNNEnc, dimRNNDec):
    W_lang_align = shared_normal(2*dimLangRNN, dimAlign)
    W_hdec_align = shared_normal(dimRNNDec, dimAlign)
    b_align = shared_zeros(dimAlign)
    v_align = shared_normal_vector(dimAlign)

    W_s_hdec = shared_normal(2*dimLangRNN, 4 * dimRNNDec)

    return W_lang_align, W_hdec_align, b_align, v_align, W_s_hdec


def build_lang_encoder_and_attention_vae_decoder(dimY, dimLangRNN, dimAlign, dimX, dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ, runStepsInt, pathToWeights=None):
    x = T.matrix() # has dimension batch_size x dimX
    y = T.matrix(dtype="int32") # matrix (sentence itself) batch_size x words_in_sentence
    y_reverse = y[::-1]
    tol = 1e-04

    if pathToWeights != None:
        W_y_hLangEnc, W_hLangEnc_hLangEnc, b_hLangEnc, W_yRev_hLangEncRev, W_hLangEncRev_hLangEncRev, b_hLangEncRev, W_lang_align, W_hdec_align, b_align, v_align, W_s_hdec, W_hdec_read_attent, b_read_attent, W_henc_henc, W_inp_henc, b_henc, W_henc_mu, W_henc_logsigma, b_mu, b_logsigma, W_hdec_hdec, W_z_hdec, b_hdec, W_hdec_write_attent, b_write_attent, W_hdec_c, b_c, W_hdec_mu_and_logsigma_prior, b_mu_and_logsigma_prior, h0_lang, h0_enc, h0_dec, c0 = load_weights(pathToWeights)
    else:
        W_y_hLangEnc, W_hLangEnc_hLangEnc, b_hLangEnc, W_yRev_hLangEncRev, W_hLangEncRev_hLangEncRev, b_hLangEncRev = create_lang_encoder_weights(dimY, dimLangRNN)
        W_lang_align, W_hdec_align, b_align, v_align, W_s_hdec = create_align_weights(dimLangRNN, dimAlign, dimRNNEnc, dimRNNDec)
        W_hdec_read_attent, b_read_attent, W_henc_henc, W_inp_henc, b_henc, W_henc_mu, W_henc_logsigma, b_mu, b_logsigma, W_hdec_hdec, W_z_hdec, b_hdec, W_hdec_write_attent, b_write_attent, W_hdec_c, b_c = create_lstm_weights(dimReadAttent, dimWriteAttent, dimRNNEnc, dimRNNDec, dimZ)
        W_hdec_mu_and_logsigma_prior = shared_normal(dimRNNDec, 2*dimZ)
        b_mu_and_logsigma_prior = shared_zeros(2*dimZ)

        h0_lang = theano.shared(np.zeros((1,dimLangRNN)).astype(floatX))
        h0_enc = theano.shared(np.zeros((1,dimRNNEnc)).astype(floatX))
        h0_dec = theano.shared(np.zeros((1,dimRNNDec)).astype(floatX))
        # initialize c0 very very small, so that sigmoid(c0) ~= 0 which is an image with black background
        c0 = theano.shared(-10*np.ones((1,dimX)).astype(floatX))

    params = [W_y_hLangEnc, W_hLangEnc_hLangEnc, b_hLangEnc, W_yRev_hLangEncRev, W_hLangEncRev_hLangEncRev, b_hLangEncRev, W_lang_align, W_hdec_align, b_align, v_align, W_s_hdec, W_hdec_read_attent, b_read_attent, W_henc_henc, W_inp_henc, b_henc, W_henc_mu, W_henc_logsigma, b_mu, b_logsigma, W_hdec_hdec, W_z_hdec, b_hdec, W_hdec_write_attent, b_write_attent, W_hdec_c, b_c, W_hdec_mu_and_logsigma_prior, b_mu_and_logsigma_prior, h0_lang, h0_enc, h0_dec, c0]

    h0_lang = T.extra_ops.repeat(h0_lang, repeats=y.shape[0], axis=0)
    h0_enc = T.extra_ops.repeat(h0_enc, repeats=y.shape[0], axis=0)
    h0_dec = T.extra_ops.repeat(h0_dec, repeats=y.shape[0], axis=0)
    c0 = T.extra_ops.repeat(c0, repeats=y.shape[0], axis=0)

    cell0_lang = T.zeros((y.shape[0],dimLangRNN))
    cell0_enc = T.zeros((y.shape[0],dimRNNEnc))
    cell0_dec = T.zeros((y.shape[0],dimRNNDec))

    kl_0 = T.zeros(())
    mu_prior_0 = T.zeros((y.shape[0],dimZ))
    log_sigma_prior_0 =  T.zeros((y.shape[0],dimZ))

    run_steps = T.scalar(dtype='int32')
    eps = rng.normal(size=(runStepsInt,y.shape[0],dimZ), avg=0.0, std=1.0, dtype=floatX)

    def recurrence_lang(y_t, h_tm1, cell_tm1, W_y_h, W_h_h, b_h):
        # Transform y_t into correct representation
        # y_t is 1 x batch_size

        temp_1hot = T.zeros((y_t.shape[0], dimY))
        y_t_1hot = T.set_subtensor(temp_1hot[T.arange(y_t.shape[0]), y_t], 1) # batch_size x dimY

        lstm_t = T.dot(y_t_1hot, W_y_h) + T.dot(h_tm1, W_h_h) + b_h
        i_t = T.nnet.sigmoid(lstm_t[:, 0*dimLangRNN:1*dimLangRNN])
        f_t = T.nnet.sigmoid(lstm_t[:, 1*dimLangRNN:2*dimLangRNN])
        cell_t = f_t * cell_tm1 + i_t * T.tanh(lstm_t[:, 2*dimLangRNN:3*dimLangRNN])
        o_t = T.nnet.sigmoid(lstm_t[:, 3*dimLangRNN:4*dimLangRNN])
        h_t = o_t * T.tanh(cell_t)

        return [h_t, cell_t]

    (h_t_forward, _), updates_forward_lstm = theano.scan(lambda y_t, h_tm1, cell_tm1, W_y_h, W_h_h, b_h: recurrence_lang(y_t, h_tm1, cell_tm1, W_y_h, W_h_h, b_h),
                                                        sequences=y.T, outputs_info=[h0_lang, cell0_lang], non_sequences=[W_y_hLangEnc, W_hLangEnc_hLangEnc, b_hLangEnc])

    (h_t_backward, _), updates_backward_lstm = theano.scan(lambda y_t, h_tm1, cell_tm1, W_y_h, W_h_h, b_h: recurrence_lang(y_t, h_tm1, cell_tm1, W_y_h, W_h_h, b_h),
                                                        sequences=y_reverse.T, outputs_info=[h0_lang, cell0_lang], non_sequences=[W_yRev_hLangEncRev, W_hLangEncRev_hLangEncRev, b_hLangEncRev])

    
    # h_t_forward is sentence_length x batch_size x dimLangRNN (example 6 x 100 x 128)
    h_t_lang = T.concatenate([h_t_forward, h_t_backward], axis=2) # was -1
    hid_align = h_t_lang.dimshuffle([0,1,2,'x']) * W_lang_align.dimshuffle(['x','x',0,1])      
    hid_align = hid_align.sum(axis=2) # sentence_length x batch_size x dimAlign # was -2

    read_attention_model = SelectiveAttentionModel(height, width, dimReadAttent)
    write_attention_model = SelectiveAttentionModel(height, width, dimWriteAttent)

    def recurrence(eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, h_tm1_enc, cell_tm1_enc, kl_tm1, mu_prior_tm1, log_sigma_prior_tm1):

        # Step 1
        x_t_hat = x - T.nnet.sigmoid(c_tm1)

        # Step 2
        read_attent_params = T.dot(h_tm1_dec, W_hdec_read_attent) + b_read_attent # dimension batch_size x 5
        g_y_read, g_x_read, delta_read, sigma_read, gamma_read = read_attention_model.matrix2att(read_attent_params)
        
        x_read = read_attention_model.read(x, g_y_read, g_x_read, delta_read, sigma_read)
        x_t_hat_read = read_attention_model.read(x_t_hat, g_y_read, g_x_read, delta_read, sigma_read)

        r_t = gamma_read * T.concatenate([x_read, x_t_hat_read], axis=1)
            
        # Step 3

        # Step new calculate alignments
        hdec_align = T.dot(h_tm1_dec, W_hdec_align) # batch_size x dimAlign
        all_align = T.tanh(hid_align + hdec_align.dimshuffle(['x', 0, 1]) + b_align.dimshuffle(['x','x',0])) # sentence_length x batch_size x dimAlign

        e = all_align * v_align.dimshuffle(['x','x',0])
        e = e.sum(axis=2) # sentence_length x batch_size # was -1

        # normalize
        alpha = (T.nnet.softmax(e.T)).T # sentence_length x batch_size

        # sentence representation at time T
        s_t = alpha.dimshuffle([0, 1, 'x']) * h_t_lang
        s_t = s_t.sum(axis=0) # batch_size x (dimLangRNN * 2)

        # no peepholes for lstm
        lstm_t_enc = T.dot(h_tm1_enc, W_henc_henc) + T.dot(T.concatenate([r_t, h_tm1_dec], axis=1), W_inp_henc) + b_henc
        i_t_enc = T.nnet.sigmoid(lstm_t_enc[:, 0*dimRNNEnc:1*dimRNNEnc])
        f_t_enc = T.nnet.sigmoid(lstm_t_enc[:, 1*dimRNNEnc:2*dimRNNEnc])
        cell_t_enc = f_t_enc * cell_tm1_enc + i_t_enc * T.tanh(lstm_t_enc[:, 2*dimRNNEnc:3*dimRNNEnc])
        o_t_enc = T.nnet.sigmoid(lstm_t_enc[:, 3*dimRNNEnc:4*dimRNNEnc])
        h_t_enc = o_t_enc * T.tanh(cell_t_enc)

        # Step 4
        mu_enc = T.dot(h_t_enc, W_henc_mu) + b_mu
        log_sigma_enc = 0.5 * (T.dot(h_t_enc, W_henc_logsigma) + b_logsigma)
        z_t = mu_enc + T.exp(log_sigma_enc) * eps_t

        kl_t = kl_tm1 + T.sum(-1 + ((mu_enc - mu_prior_tm1)**2  + T.exp(2*log_sigma_enc)) / (T.exp(2 * log_sigma_prior_tm1)) - 2*log_sigma_enc + 2*log_sigma_prior_tm1)

        # Step 5
        lstm_t_dec = T.dot(h_tm1_dec, W_hdec_hdec) + T.dot(z_t, W_z_hdec) + T.dot(s_t, W_s_hdec) + b_hdec
        i_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 0*dimRNNDec:1*dimRNNDec])
        f_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 1*dimRNNDec:2*dimRNNDec])
        cell_t_dec = f_t_dec * cell_tm1_dec + i_t_dec * T.tanh(lstm_t_dec[:, 2*dimRNNDec:3*dimRNNDec])
        o_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 3*dimRNNDec:4*dimRNNDec])
        h_t_dec = o_t_dec * T.tanh(cell_t_dec)

        # mu and logsigma depend on the activations of the hidden states of the decoder 
        mu_and_logsigma_prior_t = T.tanh(T.dot(h_t_dec, W_hdec_mu_and_logsigma_prior) + b_mu_and_logsigma_prior)
        mu_prior_t = mu_and_logsigma_prior_t[:, 0:dimZ]
        log_sigma_prior_t = mu_and_logsigma_prior_t[:, dimZ:2*dimZ]

        # Step 6
        write_attent_params = T.dot(h_t_dec, W_hdec_write_attent) + b_write_attent
        window_t = T.dot(h_t_dec, W_hdec_c) + b_c

        g_y_write, g_x_write, delta_write, sigma_write, gamma_write = write_attention_model.matrix2att(write_attent_params)
        x_t_write = write_attention_model.write(window_t, g_y_write, g_x_write, delta_write, sigma_write)

        c_t = c_tm1 + 1.0/gamma_write * x_t_write
        return [c_t.astype(floatX), h_t_dec.astype(floatX), cell_t_dec.astype(floatX), h_t_enc.astype(floatX), cell_t_enc.astype(floatX), kl_t.astype(floatX), mu_prior_t.astype(floatX), log_sigma_prior_t.astype(floatX), read_attent_params, write_attent_params]


    def recurrence_from_prior(eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, mu_prior_tm1, log_sigma_prior_tm1):
        z_t = mu_prior_tm1 + T.exp(log_sigma_prior_tm1) * eps_t

        # Step New (calculate alignment)
        hdec_align = T.dot(h_tm1_dec, W_hdec_align) # batch_size x dimAlign
        all_align = T.tanh(hid_align + hdec_align.dimshuffle(['x', 0, 1]) + b_align.dimshuffle(['x','x',0])) # sentence_length x batch_size x dimAlign

        e = all_align * v_align.dimshuffle(['x','x',0])
        e = e.sum(axis=2) # sentence_length x batch_size # was -1

        # normalize
        alpha = (T.nnet.softmax(e.T)).T # sentence_length x batch_size

        # sentence representation at time T
        s_t = alpha.dimshuffle([0, 1, 'x']) * h_t_lang
        s_t = s_t.sum(axis=0) # batch_size x (dimLangRNN * 2)

        # Step 5
        lstm_t_dec = T.dot(h_tm1_dec, W_hdec_hdec) + T.dot(z_t, W_z_hdec) + T.dot(s_t, W_s_hdec) + b_hdec
        i_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 0*dimRNNDec:1*dimRNNDec])
        f_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 1*dimRNNDec:2*dimRNNDec])
        cell_t_dec = f_t_dec * cell_tm1_dec + i_t_dec * T.tanh(lstm_t_dec[:, 2*dimRNNDec:3*dimRNNDec])
        o_t_dec = T.nnet.sigmoid(lstm_t_dec[:, 3*dimRNNDec:4*dimRNNDec])
        h_t_dec = o_t_dec * T.tanh(cell_t_dec)

        # mu and logsigma depend on the activations of the hidden states of the decoder # reference to the Philip Bachman's paper (Data generation as sequential decision making)
        mu_and_logsigma_prior_t = T.tanh(T.dot(h_t_dec, W_hdec_mu_and_logsigma_prior) + b_mu_and_logsigma_prior)
        mu_prior_t = mu_and_logsigma_prior_t[:, 0:dimZ]
        log_sigma_prior_t = mu_and_logsigma_prior_t[:, dimZ:2*dimZ]

        # Step 6
        write_attent_params = T.dot(h_t_dec, W_hdec_write_attent) + b_write_attent
        window_t = T.dot(h_t_dec, W_hdec_c) + b_c

        g_y_write, g_x_write, delta_write, sigma_write, gamma_write = write_attention_model.matrix2att(write_attent_params)
        x_t_write = write_attention_model.write(window_t, g_y_write, g_x_write, delta_write, sigma_write)

        c_t = c_tm1 + 1.0/gamma_write * x_t_write
        return [c_t.astype(floatX), h_t_dec.astype(floatX), cell_t_dec.astype(floatX), mu_prior_t.astype(floatX), log_sigma_prior_t, write_attent_params, alpha.T]

    all_params = params[:]
    all_params.append(x)
    all_params.append(hid_align)
    all_params.append(h_t_lang)

    (c_t, h_t_dec, cell_t_dec, h_t_enc, cell_t_enc, kl_t, mu_prior_t, log_sigma_prior_t, read_attent_params, write_attent_params), updates_train = theano.scan(lambda eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, h_tm1_enc, cell_tm1_enc, kl_tm1, mu_prior_tm1, log_sigma_prior_tm1, *_: recurrence(eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, h_tm1_enc, cell_tm1_enc, kl_tm1, mu_prior_tm1, log_sigma_prior_tm1),
                                                        sequences=eps, outputs_info=[c0, h0_dec, cell0_dec, h0_enc, cell0_enc, kl_0, mu_prior_0, log_sigma_prior_0, None, None], non_sequences=all_params, n_steps=run_steps)

    all_gener_params = params[:]
    all_gener_params.append(hid_align)
    all_gener_params.append(h_t_lang)

    (c_t_gener, h_t_dec_gener, cell_t_dec_gener, mu_prior_t_gener, log_sigma_prior_t_gener, write_attent_params_gener, alphas_gener), updates_gener = theano.scan(lambda eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, mu_prior_tm1, log_sigma_prior_tm1, *_: recurrence_from_prior(eps_t, c_tm1, h_tm1_dec, cell_tm1_dec, mu_prior_tm1, log_sigma_prior_tm1),
                                                        sequences=eps, outputs_info=[c0, h0_dec, cell0_dec, mu_prior_0, log_sigma_prior_0, None, None], non_sequences=all_gener_params, n_steps=run_steps)

    c_t_final = T.nnet.sigmoid(c_t[-1])
    kl_final = 0.5 * kl_t[-1]
    logpxz = T.nnet.binary_crossentropy(c_t_final,x).sum()
    log_likelihood = kl_final + logpxz
    
    log_likelihood = log_likelihood.sum() / y.shape[0]
    kl_final = kl_final.sum() / y.shape[0]
    logpxz = logpxz.sum() / y.shape[0]

    return [kl_final, logpxz, log_likelihood, c_t, c_t_gener, x, y, run_steps, updates_train, updates_gener, read_attent_params, write_attent_params, write_attent_params_gener, alphas_gener, params, mu_prior_t_gener, log_sigma_prior_t_gener]


if __name__ == '__main__':
    pass