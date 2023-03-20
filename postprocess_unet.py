import tensorflow as tf
import os

import time
import matplotlib.pyplot as plt
import numpy as np

from sys import argv, exit
from optparse import OptionParser

from utils.utils_model import fbp_convnet 
from utils.utils_tfds import load_data
from utils.utils_im import save_im

os.environ['PYTHONINSPECT'] = '1'

#plt.rcParams['figure.figsize'] = (8.0, 6.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

if (len(argv) < 2) :
  print ("python post_unet.py epochs_to_restore")
  print ('''
    -n [num epochs ] 
  ''')

  exit (1)


parser = OptionParser()

parser.add_option("-n", "--epochs", type = "int", dest="epochs", \
    help="num of c2", default=10)

(options, args) = parser.parse_args()

num_epochs = int(args[0] )

xdim = 512
cropsize = 64 

input_shape = (cropsize, cropsize, 1)
input_shape = (512, 512, 1)

# some parameter of unet
num_layers, num_channels, kernel_size = 5 , 64 , 3
#num_channels = 64 
#kernel_size =  3 
unet_parms = (num_layers, num_channels, kernel_size)
initializer = tf.keras.initializers.RandomUniform (minval = -0.01, maxval = 0.01 ) 


unet = fbp_convnet ( input_shape,  initializer = initializer, unet_parms = unet_parms ) 

print (unet.summary () )

# related to model save
checkpoint_path = "./checkpoints_unet/"
#checkpoint = tf.train.Checkpoint(optimizer=optimizer, model = unet)
checkpoint = tf.train.Checkpoint( model = unet)
manager = tf.train.CheckpointManager( checkpoint, directory=checkpoint_path, max_to_keep=5)
status = checkpoint.restore(manager.latest_checkpoint)

if status != None : 
  print ('checkpoint restored')
else : 
  print ('checkpoint not restored, reinitializing')


#fname_train =  "./fnames_train40.txt"
#fname_test =  "./fnames_test5.txt"


LDCT = False 
LDCT = True

if LDCT :

  fname_train = '/netscratch/jxu4/DL/TCIA/LDCT/train_first45_all_im.txt'
  fname_test = '/netscratch/jxu4/DL/TCIA/LDCT/test_last5_all_im.txt'
  fname_test = '/netscratch/jxu4/DL/TCIA/LDCT/random_5.txt'
  fname_test = './random_1.txt'

  num_train = 7200
  num_val =  80
  num_test = 5 
  batch_size = 1

else :

  fname_train = '/netscratch/jxu4/tumor_growth/RD/unet/denoising_train70_rcn.txt'
  fname_test = '/netscratch/jxu4/tumor_growth/RD/unet/denoising_train70_rcn.txt'

  num_train = 15000
  num_val =  500
  num_test = 1  
  batch_size =  1



train_ds, val_ds, test_ds  = load_data (fname_train, fname_test, num_train, num_val, num_test,\
         512, 512, cropsize, batch_size = batch_size)

#loss_fn = tf.keras.losses.MeanSquaredError()


xpos = 400
ypos = 263 

# y <-- F(x)  where F is the unet cnn
# dy  = F(x_0) + nabla F (x)  cdot (x - x_0)

# g1 does the following
# dx1 =  nabla F^t (x) (dy)
# g2 does the following
# dx =  nabla F (x) (dummy), ie. the forward gradient 
# (shown below)
# dx =  nabla F (x) (dx1) = nabla F(x) (something) nabla F^t (x) dy, ie. the backward, the forward prop 
# for input with FBP reconstruction, replace something by the covar of FBP recon
# not shown here

dy = np.zeros ( (1, xdim, xdim, 1) )
dy [0, ypos, xpos, 0] = 1.
dy  = tf.convert_to_tensor (dy, dtype  = tf.float32)

out_save = np.zeros ( (num_test, xdim, xdim*3) )
t = 0 
for x_np, y_np, maxval, minval, label_fname in test_ds:

    # Use the model function to build the forward pass.
    with tf.GradientTape () as g2 : 
      g2.watch (dy)  
      with tf.GradientTape () as g1 : 
        g1.watch (x_np)  
        scores = unet(x_np, training=False)
        loss = tf.reduce_mean (y_np -  scores)**2

      dx1 = g1.gradient (scores,  x_np, output_gradients = dy )

    dx = g2.gradient (dx1, dy, output_gradients = dx1 )

    unet_i =  x_np * (maxval -  minval) + minval
    unet_o =  scores * (maxval - minval) + minval
    unet_gt = y_np  # * (maxval - minval) + minval

    out_save [t, :,  0:xdim] = unet_i [0, :, :, 0].numpy()
    out_save [t, :,  xdim:2*xdim] = unet_o [0, :, :, 0].numpy()
    out_save [t, :,  2*xdim:3*xdim] = unet_gt [0, :, :, 0].numpy()
    t += 1


if False : 
  outname = '{}.im'.format (manager.latest_checkpoint )
  save_im (out_save,  outname)


wc = 1034
ww = 400

roi_h  = 32
n_samples  =  1

outim = np.concatenate ( (unet_i [0:n_samples, :, :, 0].numpy(), unet_o [0:n_samples, :, :, 0].numpy() , unet_gt [0:n_samples, :, :, 0].numpy()),  axis = -1  )  

outim_o = np.reshape (outim, ( xdim*n_samples, xdim*3) ) 

fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
ax1.imshow(outim_o , vmin = wc -ww/2, vmax = wc + ww/2) 
ax1.set_xticklabels ([])
ax1.set_yticklabels ([])
#ax1.set_title ('restored from {}'.format(os.path.basename (manager.latest_checkpoint) ))
fname =  label_fname[0].numpy() .decode ('utf-8')
patid = re.sub(r".+LDCT-+(L\d+)[-]+rc.*$", r"\1", fname.replace('/', '-')) 
ax1.set_title ('{}, {}'.format(patid, os.path.basename ( fname) ) ) 
ax1.text (0 + 2, 0 + 20, 'quarter dose', color = 'w')
ax1.text (xdim + 2, 0 + 20, 'unet out', color = 'w')
ax1.text (xdim*2 + 2, 0 + 20, 'full dose', color= 'w')
xx = [i + xdim for i in [xpos - roi_h, xpos + roi_h, xpos +roi_h, xpos - roi_h, xpos -roi_h] ] 
yy = [ypos - roi_h, ypos - roi_h, ypos +roi_h, ypos + roi_h, ypos -roi_h]
ax1.plot (xx, yy , linewidth = 1, color =  'r')
ax1.plot ([xpos + xdim], [ypos] , linestyle = None, color =  'r', marker ='+')
ax1.plot ( [xdim-0.5, xdim-0.5],  [0.5, xdim - 0.5] , linewidth = 1, color =  'w')
ax1.plot ( [2*xdim-0.5, 2*xdim-0.5],  [0.5, xdim - 0.5] , linewidth = 1, color =  'w')

plt.show (block = False)

fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
a = dx1 [0, ypos - roi_h:ypos+roi_h, xpos - roi_h:xpos+roi_h, 0].numpy()
b = dx [0, ypos - roi_h:ypos+roi_h, xpos - roi_h:xpos+roi_h, 0].numpy()
c = unet_o [0, ypos - roi_h:ypos+roi_h, xpos - roi_h:xpos+roi_h, 0].numpy()
outroi = np.concatenate ( (a, b, c), axis = -1 )  
ax2[0].imshow(c , vmin = wc -ww/2, vmax = wc + ww/2) 
ax2[0].set_title  (r'unet out: $y = F(x)$')
ax2[0].plot   ( roi_h, roi_h, marker = '+', color = 'r' ) 
ax2[1].imshow(a  ) 
ax2[1].set_title ( r'$dx = \nabla F^t(x)  dy$')
ax2[2].imshow(b  )  #  vmin = wc -ww/2, vmax = wc + ww/2) 
ax2[2].set_title ( r'$db = \nabla F(x)  dx$')
plt.show (block = False)

savefig = True
savefig = False
if savefig : 
fig2.savefig ('gradient.png', bbox_inches = 'tight', dpi = 300 )
fig1.savefig ('output.png', bbox_inches = 'tight', dpi = 300 )



#ax1.set_xlabel(' train [epochs]', color='r')
#ax1.set_yscale('log')

if False : 
  fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
  ax2.plot (range (t), train_loss)
  ax2.set_xlabel(' train [updates]', color='b')
  ax2.set_ylabel('loss', color = 'r')




