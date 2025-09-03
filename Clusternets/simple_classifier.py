import numpy as np
import scipy
import h5py as h5
import matplotlib.pyplot as plt
import pylab as plt
import pandas as pd
from scipy import interpolate
from collections import defaultdict
import seaborn as sns
Colors = sns.color_palette("colorblind", 30).as_hex()
import os
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from matplotlib.pyplot import figure
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint
import Pk_library as PKL
import psutil,os

def delta(snapshot,grid):
  import numpy as np
  import MAS_library as MASL

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


def random_cut(ltrain,lmax):
    i0,j0,k0 = np.random.randint(0,lmax-ltrain,3)
    return slice(i0,i0+ltrain),slice(j0,j0+ltrain),slice(k0,k0+ltrain)

def data_provider_train(nbatch) :

  x = [] # w = -1
  y = [] # w = -0.98
    

  for _ in range(nbatch):
      slc = random_cut(ltrain,lmax)
      slc1 = slc
      slca = random_cut(ltrain,lmax)
      slc1a = slca
      slcb = random_cut(ltrain,lmax)
      slc1b = slcb
      slcc = random_cut(ltrain,lmax)
      slc1c = slcc
      slcd = random_cut(ltrain,lmax)
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
  #x = np.expand_dims(x,axis=-1)
  #y = np.expand_dims(y,axis=-1)
  ## add new dimension to x and y
  

  return (x,y)

def data_provider_test(nbatch) :

  x = [] # w = -1
  y = [] # w = -0.98
    

  for _ in range(nbatch):
      slc = random_cut(ltrain,lmax)
      slc1 = slc


      x.append((delta6)[slc])  #seed = 42
    
      y.append((delta12)[slc1]) #seed = 46
  
  x = np.array(x)
  y = np.array(y)
  ## Convert x and y  into array
  #x = np.expand_dims(x,axis=-1)
  #y = np.expand_dims(y,axis=-1)
  ## add new dimension to x and y
  

  return (x,y)    



w_range = [7,8,85,9,95,97,98,99]

for w in w_range:

  Ngrid=256
  #w = 99
  L = 2048

  delta1 = delta(f'/mn/stornext/u3/hassanif/amir/codes/gevolution/output_wm0d{w}_s42_N256_L{L}_cs1/snap001_cdm',Ngrid) # w = -1, seed1
  delta2 = delta(f'/mn/stornext/u3/hassanif/amir/codes/gevolution/output_wm0d{w}_s43_N256_L{L}_cs1/snap001_cdm',Ngrid) # w = -1, seed1

  delta3 = delta(f'/mn/stornext/u3/hassanif/amir/codes/gevolution/output_wm0d{w}_s44_N256_L{L}_cs1/snap001_cdm',Ngrid) # w = -1, seed1
  delta4 = delta(f'/mn/stornext/u3/hassanif/amir/codes/gevolution/output_wm0d{w}_s45_N256_L{L}_cs1/snap001_cdm',Ngrid) # w = -1, seed1
  delta5 = delta(f'/mn/stornext/u3/hassanif/amir/codes/gevolution/output_wm0d{w}_s46_N256_L{L}_cs1/snap001_cdm',Ngrid) # w = -1, seed1
  delta6 = delta(f'/mn/stornext/u3/hassanif/amir/codes/gevolution/output_wm0d{w}_s47_N256_L{L}_cs1/snap001_cdm',Ngrid) # w = -1, seed1


  #wcdm
  delta7 = delta(f'/mn/stornext/u3/hassanif/amir/codes/gevolution/output_wm1_s42_N256_L{L}_cs1/snap001_cdm',Ngrid)
  delta8 = delta(f'/mn/stornext/u3/hassanif/amir/codes/gevolution/output_wm1_s43_N256_L{L}_cs1/snap001_cdm',Ngrid)

  delta9 = delta(f'/mn/stornext/u3/hassanif/amir/codes/gevolution/output_wm1_s44_N256_L{L}_cs1/snap001_cdm',Ngrid)
  delta10 = delta(f'/mn/stornext/u3/hassanif/amir/codes/gevolution/output_wm1_s45_N256_L{L}_cs1/snap001_cdm',Ngrid)

  delta11 = delta(f'/mn/stornext/u3/hassanif/amir/codes/gevolution/output_wm1_s46_N256_L{L}_cs1/snap001_cdm',Ngrid)
  delta12 = delta(f'/mn/stornext/u3/hassanif/amir/codes/gevolution/output_wm1_s47_N256_L{L}_cs1/snap001_cdm',Ngrid)

lmax = int(Ngrid)
ltrain = int(Ngrid/4)


data_train = data_provider_train(2000)
X_train = np.concatenate([data_train[0],data_train[1]],axis=0)
print('X_train : ',X_train.shape)

data_test = data_provider_test(2000)
X_test = np.concatenate([data_test[0],data_test[1]],axis=0)
print('X_test : ',X_test.shape)


mean_train = []
std_train = []
for i in range(X_train.shape[0]):
    mean_tr = np.mean(X_train[i])
    std_tr = np.std(X_train[i])
    
    mean_train.append(mean_tr)
    std_train.append(std_tr)


mean_test = []
std_test = []
for i in range(X_test.shape[0]):
    mean_te = np.mean(X_test[i])
    std_te = np.std(X_test[i])
    
    mean_test.append(mean_te)
    std_test.append(std_te)


std_test_wcdm = std_test[:2000]
std_test_Lcdm = std_test[-2000:]

std_train_wcdm = std_train[:10000]
std_train_Lcdm = std_train[-10000:]

mean_w =  np.mean(std_train_wcdm) 
mean_L = np.mean(std_train_Lcdm)

result = []
for i in range(len(std_test_wcdm)):
    a = std_test_wcdm[i]
    if np.abs(a - mean_w)<np.abs(a - mean_L):
        result.append(1)
    else:
        result.append(0)

result_L = []
for i in range(len(std_test_Lcdm)):
    b = std_test_Lcdm[i]
    if np.abs(b - mean_w)>np.abs(b - mean_L):
        result_L.append(1)
    else:
        result_L.append(0)



TP = result.count(1)
FN = result.count(0)

TN = result_L.count(1)
FP = result_L.count(0)

acc = (TP+TN) / (TP+TN+FP+FN)
print(acc)
