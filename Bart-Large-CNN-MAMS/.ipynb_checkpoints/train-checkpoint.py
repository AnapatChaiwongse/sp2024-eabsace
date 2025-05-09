import pandas as pd
import torch
import sys
from seq2seq_model_M import Seq2SeqModel

sys.path.insert(0, '/home/thanapon.nor/course-eval/Bart-Large-CNN')

fold = "1"

train_path = f"../Dataset/backtran/org-bt_all-fold{fold}.txt"
# model_path = f"outputs/original/epoch{int(fold)-1}"
model_path = "facebook/bart-large-cnn"

with open(train_path, "r",encoding='utf-8') as f:
    file = f.readlines()
train_data = []
for line in file:
    # print(line)
    x, y = line.split("\001")[0], line.strip().split("\001")[1]
    train_data.append([x, y])
    #else:
    #    print(f"Issue with line: {line.strip()}")
# train_data = [
#     ["one", "1"],
#     ["two", "2"],
# ]

train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])
#print(train_df["target_text"])
# steps = [1, 2, 3, 4, 6]
# learing_rates = [4e-5, 2e-5, 1e-5, 3e-5]
steps = [1]
learing_rates = [4e-5]


print("Training Start")
best_accuracy = 0
for lr in learing_rates:
    print(f"Current training at fold-{fold}")
    for step in steps:
        model_args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": 50,
            "train_batch_size": 16,
            "num_train_epochs": 5,
            "save_eval_checkpoints": True,
            "save_model_every_epoch": True,
            "evaluate_during_training": False,
            "evaluate_generated_text": False,
            "evaluate_during_training_verbose": True,
            "use_multiprocessing": False,
            "max_length": 150,
            "manual_seed": 42,
            "gradient_accumulation_steps": step,
            "learning_rate":  lr,
            "save_steps": 99999999999999,
        }

        torch.cuda.empty_cache()  
                
        # Initialize model
        model = Seq2SeqModel(
            encoder_decoder_type="bart",
            # encoder_decoder_name=model_path,
            encoder_decoder_name="facebook/bart-large-cnn",
            args=model_args,
        )

        # print(f"Model: {model?}")

        # Train the model
        best_accuracy = model.train_model(train_df, best_accuracy)
print(f"Model Path: {model_path}")
print(f"Train dataset: {train_path}")
