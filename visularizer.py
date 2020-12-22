import os
import numpy as np
prep_folder = r'D:\FUMPE data\FUMPE_Data'
output_folder = r'D:\L_pipe\WORK_HERE\prep_result'
filelist = [f for f in os.listdir(output_folder)]
print("Shape of Arrays:")
for name in os.listdir(prep_folder):
    if os.path.exists(os.path.join(output_folder, name + '_clean.npy')):
        numpy = np.load(os.path.join(output_folder, name + '_clean.npy'))
        print(name, '\t', numpy.shape)
        print(name, '\t', numpy[0])
        ## 4th dimension is the containers
    if os.path.exists(os.path.join(output_folder, name + '_label.npy')):
        numpy = np.load(os.path.join(output_folder, name + '_label.npy'))
        print(name, '\t', numpy.shape)
    print()