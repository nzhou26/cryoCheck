# cryoCheck
`cryocheck` is a deep learning-based cryo-EM micrograph quality assessment tool. After taken your micrographs, either in format of mrc or png, it will automatically pick out good micrographs by excluding carbon, ice-contaminated, and empty ones. 
## Architecture
TO BE COMPLETE
## Installation
1. Make sure you have [conda](https://docs.conda.io/en/latest/miniconda.html), [CUDA](https://developer.nvidia.com/cuda-toolkit)(version >= 11.0), and [cuDNN](https://developer.nvidia.com/cudnn) installed
2. Clone this repo:
```
git clone https://github.com/nzhou26/cryoCheck
```
3. Create conda environment
```
conda env create -f cryocheck_env.yml
```
3. Install dependencies from pip
```
# activate the environment just created
conda activate cryocheck

# install dependencies using pip
pip install -r requirement.txt
```
## Usage
Activate conda environment first
```
conda activate cryocheck
```
Picking out micrographs in mrc format:
```
cryocheck_infer_mrc.py /your/mrc/dir/
```
Pick out good mrc and tif so you can remove bad data
```
cleanup.py /your/good_mrc/dir/ /your/total_tif/dir/
```
## Acknowledgement
All cryo-EM micrographs data that used in training are collected at Dr. Jun He's lab in [Guangzhou Institutes of Biomedicine adn Health, Chinese Academy of Sciences](http://www.gibh.cas.cn/)

For any problems, please contact nzhou26@outlook.com