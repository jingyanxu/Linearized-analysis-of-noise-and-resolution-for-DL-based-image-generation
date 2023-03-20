# Linearized-analysis-of-noise-and-resolution-for-DL-based-image-generation
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

An online repository for the paper:

Xu, J and Noo, F. "Linearized-analysis-of-noise-and-resolution-for-DL-based-image-generation." Here is the [link](https://doi.org/10.1109/TMI.2022.3214475) to the journal site. 

The code examples and the trained models will be included soon.
The supplementary information can be accessed from [supplement.pdf](https://github.com/jingyanxu/Linearized-analysis-of-noise-and-resolution-for-DL-based-image-generation/blob/main/supplement.pdf)

An example of computing forward gradient using double backward autodiff in PyTorch can be found in [example.py](https://github.com/jingyanxu/Linearized-analysis-of-noise-and-resolution-for-DL-based-image-generation/blob/main/example.py)

**march 2023 -- I added the forward and backward gradient calculation in "postprocess_unet.py" 
The training/testing data are from LDCT on TCIA, using the 50 sets of siemens data, (quarter dose as the U-NET input, and full-dose as the label).  The first 45 patients were used for training, and the last 5 were used for testing.**

the following is a picture the result of the trained UNET, UNET input, UNET output, and label 
![input/output of trained UNET](https://github.com/jingyanxu/Linearized-analysis-of-noise-and-resolution-for-DL-based-image-generation/blob/main/output.png?raw=true)
and the result of gradient calculation
![gradient of trained UNET](https://github.com/jingyanxu/Linearized-analysis-of-noise-and-resolution-for-DL-based-image-generation/blob/main/gradient.png?raw=true)

The UNET is not the same as the one used in the paper due to an unfortunate [incident.](https://github.com/jingyanxu/Linearized-analysis-of-noise-and-resolution-for-DL-based-image-generation/blob/main/PXL_20220914_172013030.mp4) 

**If you are looking for the source code of the paper, I apologize for the delay. It has something to do with [this.](https://github.com/jingyanxu/Linearized-analysis-of-noise-and-resolution-for-DL-based-image-generation/blob/main/PXL_20220914_172013030.mp4) I am working on the replacement code. But it will take me some time. Please check back here late November.**
## Model Overview:

An in-depth paragraph about your project and overview of use.

### Dependencies

* Tested under Tensorflow (tf >= 2.0), python >= 3.6

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
```
python ./run_test.py 0
```


