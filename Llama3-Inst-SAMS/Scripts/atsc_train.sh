python ../run_model.py -mode train -model_checkpoint meta-llama/Llama-3.2-3B-Instruct \
-experiment_name atsc_check -task atsc -output_dir ../Output/llama3-instruct/coursera \
-inst_type 2 \
-id_tr_data_path ../../Dataset/instructed/coursera/coursera_train.csv \
-id_te_data_path ../../Dataset/instructed/coursera/coursera_test.csv \
-per_device_train_batch_size 1 -per_device_eval_batch_size 1 \
-learning_rate 5e-5 -num_train_epochs 8
