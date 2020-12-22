
config_submit = {'datapath':r'D:\FUMPE data\FUMPE_Data', # relative pathing 
        # D:\FUMPE data\FUMPE_Data
        'preprocess_result_path':r'D:\L_pipe\DataScienceDeck\DataScienceDeck\Testing\testing_models\Lung_Detection_1\prep_result',
        'outputfile':'prediction.csv',
        
        'detector_model':'net_detector',
        'detector_param':'./model/detector.ckpt',
        'classifier_model':'net_classifier',
        'classifier_param':'./model/classifier.ckpt',
        'n_gpu':1,
        'n_worker_preprocessing':1,
        'use_exsiting_preprocessing':False,
        'skip_preprocessing':False,
        'skip_detect':True}
config = {'stage1_data_path':'/work/DataBowl3/stage1/stage1/',
          'luna_raw':'/work/DataBowl3/luna/raw/',
          'luna_segment':'/work/DataBowl3/luna/seg-lungs-LUNA16/',
          
          'luna_data':'/work/DataBowl3/luna/allset',
          'preprocess_result_path':'/work/DataBowl3/stage1/preprocess/',       
          
          'luna_abbr':'./detector/labels/shorter.csv',
          'luna_label':'./detector/labels/lunaqualified.csv',
          'stage1_annos_path':['./detector/labels/label_job5.csv',
                './detector/labels/label_job4_2.csv',
                './detector/labels/label_job4_1.csv',
                './detector/labels/label_job0.csv',
                './detector/labels/label_qualified.csv'],
          'bbox_path':'../detector/results/res18/bbox/',
          'preprocessing_backend':'python'
         }


## Debug mode 