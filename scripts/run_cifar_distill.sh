#The running command of SCAKD contains parameter Settings.


# Same architecture
# T: wrn_40_2 ==> S: wrn_40_1
python3 train_student.py --path-t /data/SCAKD/save/models/cifar100/wrn_40_2_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s wrn_40_1 -r 0.1 -a 0.9 -b 0.1 -e 0.1 -f 1.5 --kd_T 4 \
		--save_model --experiments_dir 'tea-wrn40_2-stu-wrn_40_1/' \
		--experiments_name 'SCAKD' 
		
# T: wrn_40_2 ==> S: wrn_16_2
python3 train_student.py --path-t /data/SCAKD/save/models/cifar100/wrn_40_2_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s wrn_16_2 -r 0.1 -a 0.9 -b 0.1 -e 0.1 -f 0.5 --kd_T 4 \
		--save_model --experiments_dir 'tea-wrn40_2-stu-wrn16_2/' \
		--experiments_name 'SCAKD' 
		
# T: resnet56 ==> S: resnet20			
python3 train_student.py --path-t /data/SCAKD/save/models/cifar100/resnet56_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s resnet20 -r 0.1 -a 0.9 -b 0.1 -e 0.5 -f 0.1 --kd_T 4 \
		--save_model --experiments_dir 'tea-resnet56-stu-resnet20/' \
		--experiments_name 'SCAKD' 

# T: vgg13 ==> S: vgg8	
python3 train_student.py --path-t /data/SCAKD/save/models/cifar100/vgg13_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s vgg8 -r 0.1 -a 0.9 -b 1.0 -e 2.0 -f 0.5 --kd_T 4 \
		--save_model --experiments_dir 'tea-vgg13-stu-vgg8/' \
		--experiments_name 'SCAKD' \
		
# T: resnet110 ==> S: resnet32
python3 train_student.py --path-t /data/SCAKD/save/models/cifar100/resnet110_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s resnet32 -r 0.1 -a 0.9 -b 0.1 -e 0.1 -f 0.1 --kd_T 4 \
		--save_model --experiments_dir 'tea-resnet110-stu-resnet32/' \
		--experiments_name 'SCAKD' \
		
		
# Different architecture		
# T: resnet32x4 ==> S: ShuffleV1
python3 train_student.py --path-t /data/SCAKD/save/models/cifar100/resnet32x4_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s ShuffleV1 -r 0.1 -a 0.9 -b 1.0 -e 0.5 -f 1.5 --kd_T 4 \
		--save_model --experiments_dir 'tea-resnet32x4-stu-ShuffleV1/' \
		--experiments_name 'SCAKD' 
		
# T: resnet32x4 ==> S: ShuffleV2
python3 train_student.py --path-t /data/SCAKD/save/models/cifar100/resnet32x4_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s ShuffleV2 -r 0.1 -a 0.9 -b 1.0 -e 1.0 -f 0.5 --kd_T 4 \
		--save_model --experiments_dir 'tea-resnet32x4-stu-ShuffleV2/' \
		--experiments_name 'SCAKD' 

# T: wrn_40_2 ==> S: ShuffleV1
python3 train_student.py --path-t /data/SCAKD/save/models/cifar100/wrn_40_2_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s ShuffleV1 -r 0.1 -a 0.9 -b 1.0 -e 0.1 -f 1.0 --kd_T 4 \
		--save_model --experiments_dir 'tea-wrn40_2-stu-ShuffleV1/' \
		--experiments_name 'SCAKD' 
		
# T: wrn_40_2 ==> S: ShuffleV2	
python3 train_student.py --path-t /data/SCAKD/save/models/cifar100/wrn_40_2_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s ShuffleV2 -r 0.1 -a 0.9 -b 1.0 -e 0.1 -f 0.5 --kd_T 4 \
		--save_model --experiments_dir 'tea-wrn40_2-stu-ShuffleV2/' \
		--experiments_name 'SCAKD' 

# T: ResNet50 ==> S: MobileNetV2	
python3 train_student.py --path-t /data/SCAKD/save/models/cifar100/ResNet50_vanilla/ckpt_epoch_240.pth \
		--distill scakd \
		--model_s MobileNetV2 -r 0.1 -a 0.9 -b 1.0 -e 0.1 -f 1.0 --kd_T 4 \
		--save_model --experiments_dir 'tea-ResNet50-stu-MobileNetV2/' \
		--experiments_name 'SCAKD'
