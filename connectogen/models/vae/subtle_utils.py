# moduls
import numpy as np
import h5py
def augment_data(data_xy, axis_xy=[1,2], augment={'flipxy':0,'flipx':0,'flipy':0,'flipc':0,'flipc_segment':1}):
    if 'flipxy' in augment and augment['flipxy']:
        data_xy = np.swapaxes(data_xy, axis_xy[0], axis_xy[1])
    if 'flipx' in augment and augment['flipx']:
        if axis_xy[0] == 0:
            data_xy = data_xy[::-1,...]
        if axis_xy[0] == 1:
            data_xy = data_xy[:, ::-1,...]
    if 'flipy' in augment and augment['flipy']:
        if axis_xy[1] == 1:
            data_xy = data_xy[:, ::-1,...]
        if axis_xy[1] == 2:
            data_xy = data_xy[:, :, ::-1,...]
    if 'shiftx' in augment and augment['shiftx']>0:
        if axis_xy[0] == 0:
            data_xy[:-augment['shiftx'],...] = data_xy[augment['shiftx']:,...]
        if axis_xy[0] == 1:
            data_xy[:,:-augment['shiftx'],...] = data_xy[:,augment['shiftx']:,...]
    if 'shifty' in augment and augment['shifty']>0:
        if axis_xy[1] == 1:
            data_xy[:,:-augment['shifty'],...] = data_xy[:,augment['shifty']:,...]
        if axis_xy[1] == 2:
            data_xy[:,:,:-augment['shifty'],...] = data_xy[:,:,augment['shifty']:,...]        
    if 'shiftc' in augment and augment['shiftc']>0:
        if flipc_segment == 1:
            data_xy[:,:,:,:] = data_xy[:,:,:,::-1]
        elif flipc_segment == 2:        
        # if augment['shiftc']==1:
            nc = int(data_xy.shape[-1]/flipc_segment)
            data_xy[:,:,:,:nc] = data_xy[:,:,:,(nc-1)::-1]
            data_xy[:,:,:,-nc:] = data_xy[:,:,:,-1:(nc-1):-1]
        # if augment['shiftc']==2:
    return data_xy



def gen_augment_list(num_augment_flipxy, num_augment_flipx, num_augment_flipy,
                    num_augment_shiftx, num_augment_shifty):
    list_augments = []
    for flipxy in range(num_augment_flipxy):
        for flipx in range(num_augment_flipx):
            for flipy in range(num_augment_flipy):
                for shiftx in range(num_augment_shiftx):
                    for shifty in range(num_augment_shifty):
                        augment={'flipxy':flipxy,'flipx':flipx,'flipy':flipy,'shiftx':shiftx,'shifty':shifty}
                        list_augments.append(augment)
    return list_augments

