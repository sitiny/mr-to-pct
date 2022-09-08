import numpy as np
from monai.transforms import (
    AddChannel, Compose, LoadImage,
    ScaleIntensity, ToTensor,
)
# define patch-based inference function
# patches are 256 x 256 x 32 (i.e. 32 slices in axial/transverse plane)
def do_inference_3D(net, x, device, dims=(256,256), patch_size=(256,256,32), step_size=2):
    # pad image
    y = np.ones_like(x)
    pad_val = x[5,5,5] # find background value
    y *= 0
    in_shape = x.shape[0:2]
    z_size = patch_size[2]
    pad_dims = tuple(((d1-d2)//2, (d1-d2)-(d1-d2)//2) for d1,d2 in zip(dims,in_shape))
    pad_dims = pad_dims + ((0,z_size*2),)
    x_pad = np.pad(x, pad_dims, 'constant', constant_values=pad_val)
    y_pad = np.pad(y, pad_dims, 'constant', constant_values=0)

    num_overlaps = np.zeros(y_pad.shape)

    xptrans = Compose(
        [
            AddChannel(),
            AddChannel(),
            ToTensor(),
        ]
    )

    # iterate through patches & infer
    zz = 0
    while zz + z_size < x_pad.shape[2]:
        xp = x_pad[:,:,zz:zz+z_size]
        txp = xptrans(xp).to(device)
        output = net(txp)
        output = output[0,0,...].cpu().data.numpy()
        y_pad[:,:,zz:zz+z_size] += output
        num_overlaps[:,:,zz:zz+z_size] +=1
        zz += step_size

    y = y_pad[pad_dims[0][0]:y_pad.shape[0]-pad_dims[0][1],
              pad_dims[1][0]:y_pad.shape[1]-pad_dims[1][1],
              pad_dims[2][0]:y_pad.shape[2]-pad_dims[2][1]]
    num_overlaps = num_overlaps[pad_dims[0][0]:y_pad.shape[0]-pad_dims[0][1],
                                pad_dims[1][0]:y_pad.shape[1]-pad_dims[1][1],
                                pad_dims[2][0]:y_pad.shape[2]-pad_dims[2][1]]
    y=y/num_overlaps

    return y
