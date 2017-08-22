#!/usr/bin/env python
# -*- coding: utf-8 -*-

#...for the Operating System stuff.
import os

#...for the system stuff.
import sys

#...for parsing the arguments.
import argparse

#...for the logging.
import logging as lg

#...for the garbage collection.
import gc

#...for the MATH.
import numpy as np

from helpers import get_label_name, justMNIST

from vae import VariationalAutoencoder

#import numpy as np

def load_mnist(seed=123456):

    from tensorflow.examples.tutorials.mnist import input_data

    return input_data.read_data_sets("./mnist_data", one_hot=True)

#import data.mnist as mnist #https://github.com/dpkingma/nips14-ssl

if __name__ == '__main__':

    print("*")
    print("*==============*")
    print("* test_vae.py *")
    print("*==============*")
    print("*")

    # Parse the command line arguments.
    parser = argparse.ArgumentParser()
#    parser.add_argument("architectureFilePath",   help="Path to the architecture JSON file.")
    parser.add_argument("reloadPath",      help="Path to the model to load.", type=str)
    parser.add_argument("outputPath",      help="Path to the output folder.")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity",  action="store_true")
    parser.add_argument("-w", "--wipe",    help="Wipe all previous output",   action="store_true")
    args = parser.parse_args()

    reload_path = args.reloadPath

    ## The output path.
    output_path = args.outputPath
    #
    # Check if the output directory exists. If it doesn't, raise an error.
    if not os.path.isdir(output_path):
        raise IOError("* ERROR: '%s' output directory does not exist!" % (output_path))

    # Create the output subdirectories.

    LOG_DIR = os.path.join(output_path, "log")

    METAGRAPH_DIR = os.path.join(output_path, "out")

    PLOTS_DIR = os.path.join(output_path, "png")

    for DIR in (LOG_DIR, METAGRAPH_DIR, PLOTS_DIR):
        if os.path.isdir(DIR):
            if args.wipe:
                print("* Removing and recreating '%s'." % (DIR))
                os.system("rm -rf %s" % (DIR))
                os.mkdir(DIR)
        else:
            os.mkdir(DIR)

    print("*")

    # Set the logging level.
    if args.verbose:
        level=lg.DEBUG
    else:
        level=lg.INFO

    ## Log file path.
    log_file_path = os.path.join(LOG_DIR, 'log_test_vae.log')

    # Configure the logging.
    lg.basicConfig(filename=log_file_path, filemode='w', level=level)

    lg.info(" *=============*")
    lg.info(" * test_vae.py *")
    lg.info(" *=============*")
    lg.info(" *")
#    lg.info(" * Model setup JSON path : %s" % (architecture_file_path))
    lg.info(" * Reload path           : %s" % (reload_path))
    lg.info(" * Output path           : %s" % (output_path))
    lg.info(" *")

    #############################
    ''' Experiment Parameters '''
    #############################

    ## The number of minibatches in a single epoch.
    num_batches = 1000

    ## The number of latent variables (z).
    dim_z = 50

    ## The number of epochs through the full dataset.
    epochs = 3001

    ## The learning rate of the Adam optimizer.
    learning_rate = 3e-4

    # The L2 regularisation weight.
    l2_loss = 1e-6

    # The seed for the Random Number Generator (RNG).
    seed = 31415

    ## A list of hidden layer sizes for the generator p(x|z).
    hidden_layers_px = [ 600, 600 ]

    ## A list of hidden layer sizes for the inference model p(z|x).
    hidden_layers_qz = [ 600, 600 ]

    ####################
    ''' Load Dataset '''
    ####################

    #mnist_path = 'mnist/mnist_28.pkl.gz'
    #Uses anglpy module from original paper (linked at top) to load the dataset
    #train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy(mnist_path, binarize_y=True)

    data = load_mnist()

    x_train, y_train = data.train.next_batch(data.train.num_examples, False)

    x_valid, y_valid = data.validation.next_batch(data.validation.num_examples, False)

    x_test,  y_test  = data.test.next_batch(data.test.num_examples, False)

#    x_train, y_train = train_x.T, train_y.T
#    x_valid, y_valid = valid_x.T, valid_y.T
#    x_test,  y_test  = test_x.T,  test_y.T

    lg.debug("*-------------------------------------------------------------")
    lg.debug("* Training dataset:")
    lg.debug("*-------------------------------------------------------------")
    lg.debug("*--> x_train shape    : %s" % (str(x_train.shape)))
    lg.debug("*--> y_train shape    : %s" % (str(y_train.shape)))
    lg.debug("*")
    lg.debug("* y_train[0]")
    lg.debug(y_train[0])
    lg.debug("*")

    lg.debug("*-------------------------------------------------------------")
    lg.debug("* Validation dataset:")
    lg.debug("*-------------------------------------------------------------")
    lg.debug("*--> x_valid shape    : %s" % (str(x_valid.shape)))
    lg.debug("*--> y_valid shape    : %s" % (str(y_valid.shape)))
    lg.debug("*")

    lg.debug("*-------------------------------------------------------------")
    lg.debug("* Testing dataset:")
    lg.debug("*-------------------------------------------------------------")
    lg.debug("*--> x_test shape     : %s" % (str(x_test.shape)))
    lg.debug("*--> y_test shape     : %s" % (str(y_test.shape)))
    lg.debug("*")

    if False:
        for ix in range(10):
            label_name = get_label_name(y_train[ix])
            figure_name = "figure%06d_digit%01d" % (ix, label_name)
            lg.debug("* Making figure '%s'..." % (figure_name))
            justMNIST(x_train[ix], name=figure_name, outdir=PLOTS_DIR)

    ## The number of input features.
    dim_x = x_train.shape[1]

    ## The number of class labels.
    dim_y = y_train.shape[1]

    lg.info(" *-------------------------------------------------------------")
    lg.info(" * Input information")
    lg.info(" *-------------------------------------------------------------")
    lg.info(" * % 6d input features." % (dim_x))
    lg.info(" * % 6d class labels." % (dim_y))
    lg.info(" *")

    ######################################
    ''' Train Variational Auto-Encoder '''
    ######################################

    print("*")
    print("* Building the VAE...")
    print("*")

    VAE = VariationalAutoencoder(
                                    dim_x = dim_x,
                                    dim_z = dim_z,
                                    hidden_layers_px = hidden_layers_px,
                                    hidden_layers_qz = hidden_layers_qz,
                                    l2_loss = l2_loss
                                )

    print("*")
    print("* VAE built!")
    print("*")

    #draw_img uses pylab and seaborn to draw images of original vs. reconstruction
    #every n iterations (set to 0 to disable)

    print("*")
    print("* Resuming training...")
    print("*")

    VAE.train(x = x_train, \
              x_valid = x_valid, \
              epochs = epochs, \
              num_batches = num_batches, \
              learning_rate = learning_rate, \
              seed = seed, \
              stop_iter = 30, \
              print_every = 10, \
              draw_img = 0, \
              log_dir = LOG_DIR, \
              load_path = reload_path
             )

    print("*")
    print("* Training finished!")
    print("*")

    # Do something with the VAE...

    ## Ten test digits.
    test_x = x_test[0:10]

    ## The test digits, reconstructed.
    test_x_ = VAE.vae(test_x)

    # Make the figures of the digits and their reconstructions.
    for ix in range(10):
        label_name = get_label_name(y_test[ix])
        figure_name  = "figure%06d_digit%01d_test" % (ix, label_name)
        figure_name_ = "figure%06d_digit%01d_testreco" % (ix, label_name)
        lg.debug("* Making figure '%s'(_reco)..." % (figure_name))
        justMNIST(test_x[ ix], name=figure_name,  outdir=PLOTS_DIR)
        justMNIST(test_x_[ix], name=figure_name_, outdir=PLOTS_DIR)
