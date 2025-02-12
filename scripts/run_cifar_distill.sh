# Use this script to train your own student model.

# PKT
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet20 -r 0.1 -a 0.9 -b 30000
# SP
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet20 -r 0.1 -a 0.9 -b 3000
# VID
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet20 -r 0.1 -a 0.9 -b 1
# CRD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet20 -r 0.1 -a 0.9 -b 0.8
# SRRL
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill srrl --model_s resnet20 -r 0.1 -a 0.9 -b 1
# DKD
# python train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill dkd --model_s resnet20 -r 1 -a 0 -b 1 --dkd_alpha 1 --dkd_beta 2


python3 train_student.py --path-t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth \
        --distill kd \
        --model_s resnet20 -r 0.1 -a 0.9 -b 0 --kd_T 4 \
        --batch_size 64 --learning_rate 0.05 \
        --save_model \
        --experiments_dir 'tea-res56-stu-res20/kd/global_T/your_experiment_name' \
        --experiments_name 'fold-1' 
        
python3 train_student.py --path-t /data/goujp/xjh/CTKD-main/save/models/cifar100/resnet56_vanilla/ckpt_epoch_240.pth --distill scakd --model_s resnet20 -r 0.1 -a 0.1 -b 1.0 -e 0.5 -f 1.0 --kd_T 4 --batch_size 64 --learning_rate 0.05 --save_model --experiments_dir 'TMM/tea-res56-stu-res20/' --experiments_name 'SCAKD'