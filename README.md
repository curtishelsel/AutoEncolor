# AutoEncolor

Usage:
To see full list of options:
python autoencolor.py -h

To colorize image (resizes image to 128x128)
python autoencolor --image /path/to/image

To train on a specific network (default is classic)
python autoencolor --train --network reverse

To train on a specific on training set* (default is medium)
python autoencolor --train --mode tiny

*Dataset should be split between train, validation, and test folders with a subfolder for category*

For example: 
```
├── data
│   ├── test_full
│   │   └── faces
│   ├── train_full
│   │   └── faces
│   ├── validation_full
│       └── faces
```


