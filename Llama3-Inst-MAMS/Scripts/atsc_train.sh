python ../run_model.py -mode train -model_checkpoint ../../Llama3-Inst-SAMS/Output/llama3-instruct/coursera/atsc/meta-llamaLlama-3.2-3B-Instruct-atsc_check \
-experiment_name atsc_check -task atsc -output_dir ../Output/llama3-instruct-transfer/all/fold3 \
-inst_type 2 \
-id_tr_data_path ../../Dataset/instructed/all/all-fold3.csv \
-id_te_data_path ../../Dataset/instructed/testset/testset3.csv \
-per_device_train_batch_size 1 -per_device_eval_batch_size 1 \
-learning_rate 5e-5 -num_train_epochs 8
