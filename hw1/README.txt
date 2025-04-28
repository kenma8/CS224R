To generate the numbers for Question 1 Subproblem 1, run:
python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 \
--batch_size 100000 \ 
--eval_batch_size 10000 \
--num_agent_train_steps_per_iter 10000 \

python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Hopper.pkl \
--env_name Hopper-v4 --exp_name bc_hopper --n_iter 1 \
--expert_data cs224r/expert_data/expert_data_Hopper-v4.pkl \
--video_log_freq -1 \
--batch_size 100000 \ 
--eval_batch_size 10000 \
--num_agent_train_steps_per_iter 10000 \

To generate the numbers for Question 1 Subproblem 2, run:
python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 \
--batch_size 100000 \ 
--eval_batch_size 10000 \
--num_agent_train_steps_per_iter 10 \

python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 \
--batch_size 100000 \ 
--eval_batch_size 10000 \
--num_agent_train_steps_per_iter 100 \

python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 \
--batch_size 100000 \ 
--eval_batch_size 10000 \
--num_agent_train_steps_per_iter 1000 \

python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 \
--batch_size 100000 \ 
--eval_batch_size 10000 \
--num_agent_train_steps_per_iter 10000 \

python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 \
--batch_size 100000 \ 
--eval_batch_size 10000 \
--num_agent_train_steps_per_iter 100000 \

To generate the numbers for Question 2 Subproblem 1 run:
python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
--do_dagger \
--expert_data cs224r/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 \
--batch_size 100000 \
--eval_batch_size 5000 \
--num_agent_train_steps_per_iter 10000

python cs224r/scripts/run_hw1.py \
--expert_policy_file cs224r/policies/experts/Hopper.pkl \
--env_name Hopper-v4 --exp_name dagger_hopper --n_iter 10 \
--do_dagger \
--expert_data cs224r/expert_data/expert_data_Hopper-v4.pkl \
--video_log_freq -1 \
--batch_size 100000 \
--eval_batch_size 5000 \ 
--num_agent_train_steps_per_iter 10000

