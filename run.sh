python ./examples/gsplat/simple_trainer_difix3d.py default \
--ckpt /home/zack/Difix3D/data/test_0923_NT_less_01_70000.ply \
--target_frame_path /home/zack/Difix3D/data/test_0923_NT_less_01/transform.json \
--data_dir /home/zack/Difix3D/data/test_0923_NT_less_01 \
--result_dir /home/zack/Difix3D/data/test_0923_NT_less_01_result \
--data_factor 1 \
--packed \
--strategy.refine-stop-iter 0