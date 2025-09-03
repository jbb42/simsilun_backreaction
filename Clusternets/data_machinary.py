import numpy as np
import MAS_library as MASL
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint



def delta(snapshot,grid):
  snapshot = snapshot  #snapshot name
  grid     = grid                    #grid size
  ptypes   = [1]                     #CDM
  MAS      = 'CIC'                   #Cloud-in-Cell
  do_RSD   = False                   #dont do redshif-space distortions
  axis     = 0                       #axis along which place RSD; not used here
  verbose  = True   #whether print information on the progress

  # Compute the effective number of particles/mass in each voxel
  delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD, axis, verbose)

  # compute density contrast: delta = rho/<rho> - 1
  delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
  return delta


def random_cut(sub_box,main_box):
    i0,j0,k0 = np.random.randint(0,main_box-sub_box ,3)
    return slice(i0,i0+sub_box),slice(j0,j0+sub_box),slice(k0,k0+sub_box)

def data_provider_train(Number_of_sub_boxes) :

  x = [] # w = -1
  y = [] # w = -0.98
    

  for _ in range(Number_of_sub_boxes):
      slc = random_cut(sub_box,main_box)
      slc1 = slc
      slca = random_cut(sub_box,main_box)
      slc1a = slca
      slcb = random_cut(sub_box,main_box)
      slc1b = slcb
      slcc = random_cut(sub_box,main_box)
      slc1c = slcc
      slcd = random_cut(sub_box,main_box)
      slc1d = slcd


      x.append((delta1)[slc])  #seed = 42
      x.append((delta2)[slca]) #seed = 43
      x.append((delta3)[slcb]) #seed = 44
      x.append((delta4)[slcc]) #seed = 45
      x.append((delta5)[slcd]) #seed = 46
    
      y.append((delta7)[slc1])  #seed = 42
      y.append((delta8)[slc1a]) #seed = 43
      y.append((delta9)[slc1b]) #seed = 44
      y.append((delta10)[slc1c]) #seed = 45
      y.append((delta11)[slc1d]) #seed = 46
  
  x = np.array(x)
  y = np.array(y)
  ## Convert x and y  into array
  x = np.expand_dims(x,axis=-1)
  y = np.expand_dims(y,axis=-1)
  ## add new dimension to x and y
  

  return (x,y)

def data_provider_test(Number_of_sub_boxes) :

  x = [] # w = -1
  y = [] # w = -0.98


  for _ in range(Number_of_sub_boxes):
      slc = random_cut(sub_box,main_box)
      slc1 = slc


      x.append((delta6)[slc])   #seed = 47
      y.append((delta12)[slc1]) #seed = 47
  
  
  x = np.array(x)
  y = np.array(y)
  ## Convert x and y  into array
  x = np.expand_dims(x,axis=-1)
  y = np.expand_dims(y,axis=-1)
  ## add new dimension to x and y
  

  return (x,y)

def generate_labels(Number_of_sub_boxes,Class):
    for _ in range(Number_of_sub_boxes):  
        if Class == 0:
        label = np.zeros((Number_of_sub_boxes, 1))
        elif Class == 1:
        label = np.ones((Number_of_sub_boxes, 1))
    return label 


    

