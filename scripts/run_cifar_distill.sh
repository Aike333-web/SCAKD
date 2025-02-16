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


python3 train_student.py --path-t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
        --distill kd \
        --model_s ShuffleV1 -r 0.1 -a 0.9 -b 0 --kd_T 4 \
        --save_model \
        --experiments_dir 'tea-resnet32x4_vanilla-stu-ShuffleV1/' \
        --experiments_name 'KD' 


		
python3 train_student.py --path-t /data/goujp/xjh/CTKD-main/save/models/cifar100/resnet32x4_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s ShuffleV1 -r 0.1 -a 0.9 -b 1.0 -e 0.5 -f 1.5 --kd_T 4 \
		--save_model --experiments_dir 'tea-resnet32x4-stu-ShuffleV1/' \
		--experiments_name 'SCAKD' 
		
python3 train_student.py --path-t /data/goujp/xjh/CTKD-main/save/models/cifar100/wrn_40_2_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s ShuffleV1 -r 0.1 -a 0.9 -b 1.0 -e 0.1 -f 1.0 --kd_T 4 \
		--save_model --experiments_dir 'tea-wrn40_2-stu-ShuffleV1/' \
		--experiments_name 'SCAKD' 

python3 train_student.py --path-t /data/goujp/xjh/CTKD-main/save/models/cifar100/wrn_40_2_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s wrn_16_2 -r 0.1 -a 0.9 -b 1.0 -e 1.0 -f 1.0 --kd_T 4 \
		--save_model --experiments_dir 'tea-wrn40_2-stu-wrn16_2/' \
		--experiments_name 'SCAKD' 
		
python3 train_student.py --path-t /data/goujp/xjh/CTKD-main/save/models/cifar100/vgg13_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s vgg8 -r 0.1 -a 0.9 -b 1.0 -e 0.1 -f 0.1 --kd_T 4 \
		--save_model --experiments_dir 'tea-vgg18-stu-vgg8/' \
		--experiments_name 'SCAKD' \
		
python3 train_student.py --path-t /data/goujp/xjh/CTKD-main/save/models/cifar100/resnet56_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s resnet20 -r 0.1 -a 0.9 -b 1.0 -e 0.1 -f 0.1 --kd_T 4 \
		--save_model --experiments_dir 'tea-resnet56-stu-resnet20/' \
		--experiments_name 'SCAKD' 