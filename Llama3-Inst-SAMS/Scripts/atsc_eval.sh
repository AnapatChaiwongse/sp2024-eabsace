python ../run_model.py -mode eval -model_checkpoint ../Output/llama3-instruct/coursera/atsc/meta-llamaLlama-3.2-3B-Instruct-atsc_check \
-experiment_name atsc_check -task atsc -output_path ../Output \
-inst_type 2 \
-id_tr_data_path ../../Dataset/instructed/original/org-fold2.csv \
-id_te_data_path ../../Dataset/instructed/coursera/coursera_test.csv \
-per_device_train_batch_size 1 -per_device_eval_batch_size 1 \
-learning_rate 5e-5 -num_train_epochs 8 \
-max_token_length 8192