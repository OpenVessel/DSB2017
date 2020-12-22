###
### IMPORTS & calls

from preprocessing import step1_python

#####
##### CONFIGURATION
#####
pathway_1 = '/content/drive/My Drive/OpenVessel LLC/Technical /Data Science/Datasets/FUMPE_DATA/Patient_data/PAT001'
pathway_2 = r'D:\Lung CT\SPIE-AAPM Lung CT Challenge\CT-Training-BE001\01-03-2007-16904-CT INFUSED CHEST-143.1\4-HIGH RES-47.17'

test, bw1, bw2, spacing = step1_python(pathway_2)

print(test)
print(type(test))


### Full Preprocessing convert DICOM to numpy array






