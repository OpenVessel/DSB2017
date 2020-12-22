
import sys
import torch
# insert at 1, 0 is the script path (or '' in REPL)
# NEEDS relative paths
# sys  improve pathing
# requirement is folder "preprocessing_for_models" 
sys.path.insert(1, r'D:\L_pipe\DataScienceDeck\DataScienceDeck\Testing')
#from preprocessing.preprocessing import full_prep_mc,full_prep_no_mc, preprocess_muilthread
from preprocessing import full_prep_mc,full_prep_no_mc
#from ..preprocessing import preprocessing

sys.path.insert(1, r'D:\L_pipe\DataScienceDeck\DataScienceDeck\Testing\testing_models\Lung_Detection_1')
print(sys.path)
from multiprocessing import Process, freeze_support
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from config_submit import config_submit

from data_detector import DataBowl3Detector, collate
#from data_classifier import DataBowl3Classifier
#from __future__ import division

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import os



if __name__ == "__main__":
    ####################################################################### DEFINE STATIC VARIABLES ############################
    datapath = config_submit['datapath']
    prep_result_path = config_submit['preprocess_result_path']
    skip_prep = config_submit['skip_preprocessing']
    skip_detect = config_submit['skip_detect']

    ####################################################################### PREPROCESSING #########################################
    if not skip_prep:
        testsplit = full_prep_no_mc(datapath, prep_result_path,
                            n_worker = config_submit['n_worker_preprocessing'],
                            use_existing=config_submit['use_exsiting_preprocessing']) # doesn't use multiprocessing
    else:
        testsplit = os.listdir(datapath)

    ####################################################################### LOAD PRETRAINED NET DETECTOR MODEL ############################

    # why not use an import statement?
    nodmodel = import_module(config_submit['detector_model'].split('.py')[0]) # reference to net_detector.py

    config1, nod_net, loss, get_pbb = nodmodel.get_model()
    checkpoint = torch.load(config_submit['detector_param']) # net detector weights
    nod_net.load_state_dict(checkpoint['state_dict'])

    ######################################################### GPU CONFIGURATION ##########################################################
    torch.cuda.set_device(0)
    nod_net = nod_net.cuda()
    cudnn.benchmark = True
    nod_net = DataParallel(nod_net) ## Set number of GPU 

    ######################################################### LOAD DATA and TEST MODEL ################################################################
    print(f'--- skip_detect = {skip_detect} ---')
    if not skip_detect:
        print('testing detection model')

        ################### STATIC VARS ################
        margin = 32
        sidelen = 144
        config1['datadir'] = prep_result_path
        bbox_result_path = './bbox_result'
        if not os.path.exists(bbox_result_path):
            os.mkdir(bbox_result_path)

        ################### LOAD TESTING DATASET (being conscious of GPUs) ################

        # splitting and combining numpy array
        split_comber = SplitComb(side_len = sidelen,
                                max_stride = config1['max_stride'],
                                stride = config1['stride'],
                                margin = margin,
                                pad_value = config1['pad_value'])

        dataset = DataBowl3Detector(split = testsplit,
                                    config = config1,
                                    phase = 'test',
                                    split_comber = split_comber)


        # https://www.journaldev.com/36576/pytorch-dataloader
        # raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) windows
        test_loader = DataLoader(dataset,
                                batch_size = 1,
                                shuffle = False,
                                num_workers = 1,
                                pin_memory=False,
                                collate_fn =collate)

        ########################## TEST MODEL ##################################

        ## Test detect for loops the DataLoader class and saves numpy features 
        test_detect(data_loader= test_loader, 
                    net = nod_net, 
                    get_pbb = get_pbb, 
                    save_dir = bbox_result_path,
                    config = config1,
                    n_gpu=config_submit['n_gpu'])
    else:
        print('skipping testing detection model')
