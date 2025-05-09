python ../run_model.py -mode eval -model_checkpoint ../Output/llama3-instruct-transfer/all/fold3/atsc/....Llama3-Inst-SAMSOutputllama3-instructcourseraatscmeta-llamaLlama-3.2-3B-Instruct-atsc_check-atsc_check \
-experiment_name atsc_check -task atsc -output_path ../Output \
-inst_type 2 \
-id_tr_data_path ../../Dataset/instructed/original/org-fold3.csv \
-id_te_data_path ../../Dataset/instructed/testset/testset3.csv \
-per_device_train_batch_size 1 -per_device_eval_batch_size 1 \
-learning_rate 5e-5 -num_train_epochs 8
