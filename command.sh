python main.py pretrain --config \
/data1/zengyq/SkeleBYOL/config/pretrain/pretrain_skeletonbyol.yaml

tensorboard --logdir=/data1/zengyq/SkeleBYOL/work_dir/ntu60xview/

python main.py linear_evaluation \
--config ./config/linear_eval/linear_eval_skeletonbyol.yaml \
--weights /data1/zengyq/SkeleBYOL/work_dir/ntu60xview/modellr002/epoch300_model.pt

tensorboard --logdir=/data1/zengyq/SkeleBYOL/work_dir/linear_eval

python main.py finetune_evaluation \
--config config/fine_tune/fine_tune_skeletonbyol.yaml \
--weights /data1/zengyq/skeleton/work_dir/ntu60xview/lr0.1bs256m999/epoch300_model.pt

lr 005 best