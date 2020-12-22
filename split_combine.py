import torch
import numpy as np

#https://d2l.ai/chapter_convolutional-neural-networks/channels.html
#ESSENTIAL OVERVIEW:
# After the network is trained, the entire lung scans could
# be used as input to obtain all suspicious nodules. Because
# the network is fully convolutional, it is straightforward to do
# this. But it is infeasible with our GPU memory constraint.
# Even though the network needs much fewer memory in testing
# than in training, the requirement still exceeds the maximum
# memory of the GPU. To overcome this problem, we split the
# lung scans into several parts (208 x 208 x 208 x 1 per part),
# process them separately, and then combine the results. We
# keep these splits overlapped by a large margin (32 pixels)
# to eliminate the unwanted border effects during convolution
# computations.

class SplitComb():
    def __init__(self,side_len,max_stride,stride,margin,pad_value):
        self.side_len = side_len #144  
        self.max_stride = max_stride #mentions somewhere stride = 2; 16
        self.stride = stride #4 
        #The nodule targets are not necessarily located at the center of the patch
        #but had a margin larger than 12 pixels from the boundary of
        #the patch (except for a few nodules that are too large).
        self.margin = margin  #32 
        # keep these splits overlapped by a large margin (32 pixels)
        # to eliminate the unwanted border effects during convolution computations
        self.pad_value = pad_value #170
        
    def split(self, data, side_len = None, max_stride = None, margin = None):
        #data = ??, unsure of what the data being passed into this function is; 
        #seems to be 
        #cant find the split call on "split_comber" in net_detector
        #why can side_len, max_stride, margin, be set equal to None?? its because they can take their values from the _init_ method
        if side_len==None: 
            side_len = self.side_len
        if max_stride == None: 
            max_stride = self.max_stride
        if margin == None: 
            margin = self.margin 
        
        assert(side_len > margin) # True #why does side_len have to be greater than margin?? 
        assert(side_len % max_stride == 0) # 0 #side_len should be evenly divisible by max_stride
        #therefore sayi)ng cannot stride by some number that is not evenly 
        assert(margin % max_stride == 0) # 0 

        #shape BEFORE padding: (1,519,201,279)
        
        splits = []
        #_ = ghost variable
        _, z, h, w = data.shape #should be 4D already
        #ceiling function
        #below are ratio's of shape to the side_len variable passed into the _init_ function 
        nz = int(np.ceil(float(z) / side_len)) #height #4 
        nh = int(np.ceil(float(h) / side_len)) #length #2 
        nw = int(np.ceil(float(w) / side_len)) #width #2 
        
        
        nzhw = [nz,nh,nw]
        self.nzhw = nzhw

        pad = [[0, 0],
                [margin, nz * side_len - z + margin],
                [margin, nh * side_len - h + margin],
                [margin, nw * side_len - w + margin]] 

       


        # padding the data?? not sure what these numbers do to the end result 
        # pad the array (therefore data is an array), 
        # pad_width = pad (Number of values padded to the edges of each axis. 
        # ((before_1, after_1), … (before_N, after_N)) unique pad widths for each axis. 
        # ((before, after),) yields same before and after pad for each axis. (pad,) 
        # or int is a shortcut for before = after = pad width for all axes.)
        #"edge" - pads with the edge values of an array

        #NOTE: 
            #this website provides a lower level view of what np.pad is doing:
            #https://www.geeksforgeeks.org/numpy-pad-function-in-python/
        
        #change the values in data to int
        data = np.pad(data, pad, 'edge') 

        #shape AFTER padding: (1,640,352,352)

        for iz in range(nz): # 4 
            for ih in range(nh): # 2
                for iw in range(nw): # 2
                    sz = iz * side_len  #start z 
                    ez = (iz + 1) * side_len + 2 * margin #end z
                    sh = ih * side_len #start h 
                    eh = (ih + 1) * side_len + 2 * margin #end h
                    sw = iw * side_len #start w 
                    ew = (iw + 1) * side_len + 2 * margin #end w 
                    #this is the original data split by the start and end points for each axis/dimension
                    #split = data[np.newaxis,:, 0:208, 0:208, 144:352]
                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew] 
                    splits.append(split) #all of the "split" variables

        #splits now equals the combination of all of the splits (16 seperate lists), joined on the "0" axis??
        splits = np.concatenate(splits, 0)
        
        return splits, nzhw
#TEST:
#x = SplitComb(144,16,4,32,170)
#data = np.load(r"C:\Users\12673\Desktop\Projects\OpenVessel\DSB2017-master\prep_result\PAT001_clean.npy")
#print(data.shape)
#for i in range(len(x.split(data))):
#    print(i, x.split(data)[i].shape) --> prints (1,208,208,208) 16 times which is what I expected


    def combine(self, output, nzhw = None, side_len=None, stride=None, margin=None):
        
        if side_len==None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if nzhw is None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz,nh,nw = nzhw

        assert(side_len % stride == 0)  # if this stride doesnt divide side_len, error raised
        assert(margin % stride == 0) #if stride doesnt divide margin, error raised

        #these two lines change the variable values
        side_len /= stride #now side_len = 36
        margin /= stride #now margin = 8

        splits = []
        #range(len(output)) should be 16 if the output is a split function array
        for i in range(len(output)): #output seems to be a list passed to the function
            #simply appending the values in the output parameter to this new list "splits"
            splits.append(output[i]) 
        

        
        #np.ones = returns an array of given shape filled with ones
        
        #set the output array to ones for classification step later on?
        #simply making an array with the desired shape, the ones seem to be irrelevant, simply placeholders
        output = -1000000 * np.ones((
            nz * side_len,
            nh * side_len,
            nw * side_len,
            splits[0].shape[3], #takes the 3rd dimension of the shape of the first index of splits
            #EX: if splits[0].shape = (1,2,3,4), splits[0].shape[3] == 4
            splits[0].shape[4]), np.float32)
        print(output)
        print(output.shape)
        
        #idx ranges from 0-15 
        #possible values for sz,sh,sw are 0,36,72,108
        #possible values for ez,eh,ew are 36,72,108,144 
        idx = 0
        for iz in range(nz): #4 
            for ih in range(nh): #2 
                for iw in range(nw): #2 
                    
                    sz = iz * side_len 
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    #only thing that is changing here is [idx], everything inside the list seems to remain the same throughout
                    # idx --> being in range from 0-15 makes me think it is part of the 16 different input 
                    #assigns the iterations of splits (0-15) to a new variable split, indexed by the margins and side_len
                    split = splits[idx][margin:margin + side_len, margin:margin + side_len, margin:margin + side_len] #all values should be 8:44
                    
                    #this output indexing, is now equal to the split variable above
                    output[sz:ez, sh:eh, sw:ew] = split
                
                    idx += 1
        
        return output

