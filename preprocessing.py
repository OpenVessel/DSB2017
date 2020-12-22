import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from pydicom import dcmread
from skimage import measure, morphology

from scipy.io import loadmat
import h5py
from scipy.ndimage.interpolation import zoom
from skimage import measure
import warnings
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from multiprocessing import Pool
from functools import partial
import warnings
from concurrent.futures import ProcessPoolExecutor

print("DataScienceDeck.preprocessing.preprocessing")

## Input



import os 
import matplotlib.pyplot as plt
import numpy as np
from global_config_submit import ROOT_DIR
#python -m DataScienceDeck.Visualization.dev_script
def save_labels(label, root_dir=ROOT_DIR):
    print("root dir-", root_dir)
    path = os.getcwd()
    print("current directory" , path)
    global_base_path = os.path.abspath(os.path.dirname(__file__))
    
    output_1 = r'D:\L_pipe\DataScienceDeck\DataScienceDeck\Visualization\ouput_images\output_1'
    output_2 = r'D:\L_pipe\DataScienceDeck\DataScienceDeck\Visualization\ouput_images\output_2'
    numpy_output = r'DataScienceDeck\Visualization\output_numpy_labels'
    
    
    print(label.shape[0])
    
    ## save label as numpy array 
    file_path = os.path.join(global_base_path, numpy_output)
    if not os.path.exists(file_path):
        print("making prep_folder")
        os.mkdir(file_path)
    file_path = file_path + r'\\saved_data.npy'
    np.save(file_path, label)

    for i in range(label.shape[0]):
        plt.imshow(label[i])
        #x = str(ouput_dir_results + '\\' + 'images_' + i)
        plt.savefig(output_1 + '\\' + 'images_' + str(i))
    return 


def load_scan(path):
    print("Loading scan", path)
    slices = [dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2;
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num+1;
        slice_num = int(len(slices) / sec_num)
        slices.sort(key = lambda x:float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key = lambda x:float(x.ImagePositionPatient[2]))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def load_scan_deployment(dicom_list):
    ## all slices are put into the array
    slices.sort(key = lambda x: int(x.InstanceNumber))
    ##
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(slices):
    print("get pixels converted to HU")
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    
    
    # Since the scanning equipment is cylindrical in nature and image output is square,
    # we set the out-of-scan pixels to 0
    #image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    #line replacement for some reason https://github.com/lfz/DSB2017/issues/72
    #return np.array(image, dtype=np.int16), np.array([slices[0].SliceThickness] + slices[0].PixelSpacing, dtype=np.float32)

    ### returns and spacinginformation
    save_labels(image)
    return np.array(image, dtype=np.int16), np.array([slices[0].SliceThickness, slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]], dtype=np.float32)


def binarize_per_slice(image, spacing, intensity_th= -600, sigma= 1, area_th= 30, eccen_th= 0.99, bg_patch_size= 10):
    #creates the black-white dataframe as the same size of the which is zeroed so that everything in the dataframe is False, 
    #when we determine that a datapoint is valid we can declare that point true and it will stand out as a black figure 
    
    bw = np.zeros(image.shape, dtype=bool)
    #print("np.zeros(image.shape, dtype=bool)=",bw)
    # process up to the for loop creates a mask, with all corner values set to nan 
    # (because CT scanner creates round images?)
    #image.shape= (80, 512)
    image_size = image.shape[1]
    print("binarization")
    print("image.shape[1] =",image_size)
    print("image.shape[0]=", image.shape[0])
    #one axis that is a spread from -255.5 to 255.5 

    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    #print("grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)=", grid_axis)
    #print("grid_axis.shape=",grid_axis.shape)
    #uses grid_axis to mesh 2 axises of grid_axis together to create a matrix 
    x, y = np.meshgrid(grid_axis, grid_axis)
    #print("np.meshgrid(grid_axis, grid_axis) =", x,y)
    #r^2 = x^2 + y^2, d= radius of the image?, makes all of the -255.5 to 255.5 values positive 
    #but this might try to keep this -255.5 to 255.5 spread's relationship to each other 
    d = (x**2+y**2)**0.5
    #print("(x**2+y**2)**0.5 =",d)
    #Anything that is less than 206 is kept as it's float value 
    #   but anything greater (the corners) are not read (are = 0) and set to nan (null)
    nan_mask = (d<image_size/2).astype(float)
    #print("nan_mask = (d<image_size/2).astype(float) = ",nan_mask)
    nan_mask[nan_mask == 0] = np.nan
    #In short, a nan_mask matrix was created where values of (((-image_size/2+0.5)**2)+((image_size/2+0.5)**2))**0.5 
    #   have to be less than image_size/2 
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice (??) before Gaussian filtering
        # parameters for np.unique? 
        #checking to see whether the corners of the image are identical by using a 0:bg_patch_size (10x10) block 
        #and seeing whether all the values within that block are identical (# of unique pixel values within block are = 1)
        #if all the pixel values within that corner are identical then it going to apply the nan mask to that shape 
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
            
        else: 
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th 
            
        # select proper components
        #labels current_bw 
        label = measure.label(current_bw)
        #uses labels to extract properties using regionprops 
        properties = measure.regionprops(label) 
        #creates a set for valid labels to be distinguished 
        valid_label = set() 
        #for every region 
        for prop in properties:
            #eccen_th (eccentricity threshold) = 0.99, 1 is a line and 0 is a circle, 
            #    we want to keep what is not a line (prop.eccentricity < eccen_th), but we can make the threshold lower to have less detail 
            #spacing[1] & spacing[2] == 0.619141.     spacing[1] * spacing[2] == 0.383335578.  
            #    prop.area * 0.383335578 > 30 (arbitrary? area_th) so: prop.area needs to be > 78.2604113 to be valid. 
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label) 
        #np.in1d purpose is to check between two arrays and return a boolean output 
        #       of whether for each element in the 2nd array, it is there in the 1st array. 
        #current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        current_bw= np.isin(label, list(valid_label)).reshape(label.shape)
         #Effectively creating our true/false delimited bw picture by setting each of these true/false arrays equal
         #       to each row of the full bw picture frame 
        bw[i] = current_bw 

    
    return bw

#get my code in here 
def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    #save_labels(bw)
    print("all_slice_analysis")

    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)

    # remove components access to corners
    mid = int(label.shape[2] / 2)
    ## background label 0 is background information
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    print("background label - all_slice")
    for l in bg_label:
        label[label == l] = 0
    
    # select components based on volume # loop through each regions 
    print("Volume - all_slice")
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            
            label[label == prop.label] = 0

    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
    
    
    #plt.imshow(valid_label[120])
    #plt.show()
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    
    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
    
    return bw, len(valid_label)

def fill_hole(bw):
    print("fill_hole")
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)
    
    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):    
    print("two_lung_only running")
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area)*cover:
                sum = sum+area[count]
                count = count+1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter
           
        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label==properties[0].label

        return bw
    
    def fill_2d_hole(bw):
        print("fill_2d_hole")
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw
    
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw

def step1_python(case_path):
    case = load_scan(case_path)
    case_pixels, spacing = get_pixels_hu(case)
    bw = binarize_per_slice(case_pixels, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing



def process_mask(mask):
    print("Process mask")
    convex_mask = np.copy(mask)
    #print(convex_mask)
    for i_layer in range(convex_mask.shape[0]):
        ## the mask is converted to contiguous array which is just a unbrroken block of memory
        mask1  = np.ascontiguousarray(mask[i_layer])

        ## Here we apply convex_hull_image if mask is greater than zero 
        ## Convex hull is a set of pixels included in hte smallest convex polygon that surrounds 
        ## all whitte pixels in the input image
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2

    ## Generae binary structure for only morphological operations (rank, connectivity)
    ## limited to 3 dimsion 
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 

    return dilatedMask

def lumTrans(img):
    ## Raw Matrix is clipped with window of [-1200, 600] 
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def resample(imgs, spacing, new_spacing,order = 2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

def savenpy(filelist,prep_folder,data_path,use_existing=True):      
    '''
        main preprocessing function
    '''
    resolution = np.array([1,1,1])

    ## name of file PAT001 or 0912090499014
    for i in range(len(filelist)):
        name = filelist[i]

        ## does tthe prep_folder exist? 
        if use_existing == False:
            print(os.path.exists(os.path.join(prep_folder,name+'_label.npy')))
            print(os.path.exists(os.path.join(prep_folder,name+'_clean.npy')))
            if os.path.exists(os.path.join(prep_folder,name+'_label.npy')) and os.path.exists(os.path.join(prep_folder,name+'_clean.npy')):
                print(name + ' had been done')
                continue
                

            try:
                ### Step1 of preprocessing 
                #im, m1, m2, spacing = step1_python(os.path.join(data_path,name))

                case_path = os.path.join(data_path,name)
                case = load_scan(case_path)
                case_pixels, spacing = get_pixels_hu(case)
                bw = binarize_per_slice(case_pixels, spacing)
                ## bw black white
                ### some variables 
                flag = 0
                cut_num = 0
                cut_step = 2

                bw0 = np.copy(bw)
                print(bw.shape[0])
                while flag == 0 and cut_num < bw.shape[0]:
                    bw = np.copy(bw0)
                    ### ??
                    bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
                    cut_num = cut_num + cut_step
            
                bw = fill_hole(bw)
                bw1, bw2, bw = two_lung_only(bw, spacing)
                
                im = case_pixels
                m1 = bw1
                m2 = bw2
                ## Next required parameters are spacing 
                ## can be put this into a function?
                ## we make our mask by addition of two mask then muiltply by spacing and divided by reslution
                Mask = m1+m2
                print("Masking shape", Mask.shape)

                newshape = np.round(np.array(Mask.shape)*spacing/resolution) ##???
                print(Mask[0])
                xx,yy,zz= np.where(Mask) ## mask is used to select xx,yy,zz locations  

                print(xx.shape)
                print(yy.shape)
                print(zz.shape)


                ## A box is generated with minial value ranges and maxium value ranges
                box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
                box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1) ## expand the dimsions via spacing and resolutions
                box = np.floor(box).astype('int') ### TReturns the floor of the scalar x is the largestt intteger i floor removes he decamical points afterwards
                margin = 5 ## why is margin set to 5?

                ## Extendedbox 
                ## Extendbox is vstack, that is used to concaentate along the first axis of 1-D arrays of shape. 
                ## vstack is used for 3D dimensions like pixel-data with a heightt first axis, width second axis, and color as a third
                extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
                extendbox = extendbox.astype('int')


                ## Process_mask???
                    ## convex mask outline of surface curved 
                convex_mask = m1

                    ## we process both mask separate again 
                dm1 = process_mask(m1)
                dm2 = process_mask(m2)
                    # the mask are dilated in process_mask() 

                ## we created 3 different mask 
                dilatedMask = dm1+dm2
                Mask = m1+m2
                extramask = dilatedMask ^ Mask

                ## HU Values 
                bone_thresh = 210
                pad_value = 170

                im[np.isnan(im)]=-2000

                ### LumTrans ?????
                sliceim = lumTrans(im)
                ##multiplied full mask and padded 170 uint*
                sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
                ## bones and intensity of tissue
                bones = sliceim*extramask>bone_thresh
                sliceim[bones] = pad_value

                ### Resample call ???
                sliceim1,_ = resample(sliceim,spacing,resolution,order=1)

                ### Extend box ??
                sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                            extendbox[1,0]:extendbox[1,1],
                            extendbox[2,0]:extendbox[2,1]]
                            
                sliceim = sliceim2[np.newaxis,...]
                print(sliceim.shape)
                ###  numpying saving?
                ###      'preprocess_result_path':'./prep_result/',
                np.save(os.path.join(prep_folder,name+'_clean'),sliceim)
                np.save(os.path.join(prep_folder,name+'_label'),np.array([[0,0,0,0]])) #https://github.com/lfz/DSB2017/issues/6
            except:
                print('bug in '+name)
                raise
        print(name+' done')


    ## Full_prep_mc is making multiprocessing calls for pools on numpy arrays 
    ## File structure is bad
def full_prep_mc(data_path, prep_folder, n_worker = None, use_existing=True):

    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        print("making prep_folder")
        os.mkdir(prep_folder)
    
    print('starting preprocessing (with mc)')
    pool = Pool(n_worker)
    filelist = [f for f in os.listdir(data_path)]
    print(filelist)
    partial_savenpy = partial(savenpy,
                            filelist=filelist,
                            prep_folder=prep_folder,
                            data_path=data_path,
                            use_existing=use_existing)
    N = len(filelist)
    _ = pool.map(partial_savenpy,range(N))
    pool.close()
    pool.join()
    print('end preprocessing')
    return filelist


def full_prep_no_mc(data_path, prep_folder, n_worker = None, use_existing=True):
    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        print("making prep_folder")
        os.mkdir(prep_folder)

    print('starting preprocessing (without mc)')
    filelist = [f for f in os.listdir(data_path)]
    print(filelist)
    savenpy(filelist=filelist,
            prep_folder=prep_folder,
            data_path=data_path,
            use_existing=use_existing)

    print('end preprocessing')
    return filelist

def preprocess_muilthread(data_path,prep_folder,n_worker = None,use_existing=True):
    with ProcessPoolExecutor() as executor:
        # for loop goes here
    
        if not os.path.exists(prep_folder):
            print("making prep_folder")
            os.mkdir(prep_folder)

                
        print('starting preprocessing')
        ## list of all patients 
        filelist = [f for f in os.listdir(data_path)]
        print(filelist)
        executor.map(savenpy(filelist=filelist,prep_folder=prep_folder,
        data_path=data_path,use_existing=use_existing), filelist)


    print('end preprocessing')
    return filelist


def full_preprocessing(path):
    case = load_scan(path)
    case_pixels, spacing = get_pixels_hu(case)
    bw = binarize_per_slice(case_pixels, spacing)
    
    print(bw)
    
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing
