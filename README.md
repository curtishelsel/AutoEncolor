# AutoEncolor

Directoru Structure

├── data \
│   ├── test\_full\
│   ├── test\_medium\
│   ├── test\_small\
│   ├── test\_tiny\
│   ├── train\_full\
│   ├── train\_medium
│   ├── train\_small
│   ├── train\_tiny
│   ├── validation\_full
│   ├── validation\_medium
│   ├── validation\_small
│   └── validation\_tiny
├── figures
├── models
│   ├── best\_classic\_model.pt
│   ├── best\_deep\_model.pt
│   ├── best\_pooling\_model.pt
│   ├── best\_reverse\_model.pt
│   ├── current\_classic\_checkpoint.pt
│   ├── current\_deep\_checkpoint.pt
│   ├── current\_pooling\_checkpoint.pt
│   └── current\_reverse\_checkpoint.pt
├── README.md
└── src
    ├── autoencolor.py
    ├── checkpoint.py
    ├── colorizationdataset.py
    ├── colorize.py
    ├── earlystopping.py
    ├── imageloader.py
    ├── models.py
    ├── parser.py
    └── train.py


