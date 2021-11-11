# UTRAD
UTRAD for nueral networks
## Installation
This repo was tested with Ubuntu 16.04/18.04, Pytorch 1.5.0
## Running 
1. Fetch the Mvtec datasets, and extract to datasets/
2. Run training by using command:
```
python main.py --dataset_name grid
```
where --dataset_name is used to specify the catogory.

3. Validate with command:
```
python valid.py --dataset_name grid
```
4. Validate with unaligned setting:
```
python valid.py --dataset_name grid --unaligned_test
```
