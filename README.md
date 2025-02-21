# Similarity-Based Contrastive Augmented Knowledge Distillation for Visual Recognition ( SCAKD )



## Requirements 

- Python 3.8
- Pytorch 1.11.0
- Torchvision 0.12.0

## Running

1. Download the pre-trained teacher models and put them to `./save/models`.

|  Dataset | Download |
|:---------------:|:-----------------:|
| CIFAR teacher models   | [[Baidu Cloud](https://pan.baidu.com/s/1wqqKjlGC4ewvqyNgWZNRGA?pwd=82f)] |

If you want to train your teacher model, please consider using `./scripts/run_cifar_vanilla.sh`

After the training process, put your teacher model to `./save/models`.

2. Training on CIFAR-100:
- Download the dataset and change the path in `./dataset/cifar100.py line 27` to your current dataset path.
- Modify the script `scripts/run_cifar_distill.sh` according to your needs.
- Run the script.
    ```
    sh scripts/run_cifar_distill.sh  
    ```

If you have any questions, you can submit an [issue](https://github.com/Aike333-web/SCAKD/issues) on GitHub.

