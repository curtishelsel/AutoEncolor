# This class gathers all the command line
# arguments and provides them as a single
# parameter set

import argparse

class Argparse():
    def __init__(self):
        
        parser = argparse.ArgumentParser()

        parser.add_argument('-b', '--batch_size', 
                            help='sets batch size for dataloader and training',
                            type=int, default=10)

        parser.add_argument('-c','--continue_training', 
                            help='continues training the model \
                            with existing checkpoint',
                            action='store_true')

        parser.add_argument('-e', '--epochs', 
                            help='sets number of epochs to train model',
                            type=int, default=500)

        parser.add_argument('-i', '--image', 
                            help='image path to colorize',
                            type=str,
                            default=None)

        parser.add_argument('-l', '--learning_rate', 
                            help='sets learning rate for training model',
                            type=float, default=0.001)

        parser.add_argument('-m', '--mode', 
                            help='sets dataset for training model \
                            full = 70000, small = 7000, and tiny = 700',
                            choices=['full', 'small', 'tiny'], 
                            default='full')

        parser.add_argument('-n', '--network', 
                            help='sets network model for training \
                            and inference',
                            choices=['reverse', 'pooling', 'classic'], 
                            default='classic')

        parser.add_argument('-p', '--show_plot', 
                            help='shows training plot, saves otherwise',
                            action='store_true')

        parser.add_argument('-s', '--early_stopping', 
                            help='use early stopping',
                            action='store_true')

        parser.add_argument('-t','--train', 
                            help='sets program to train new model',
                            action='store_true')

        parser.add_argument('-v', '--validation', 
                            help='use validation set',
                            action='store_true')

        self.args = parser.parse_args()

    
