import pandas as pd
import torch
import sys
from seq2seq_model_S import Seq2SeqModel

sys.path.insert(0, '/home/thanapon.nor/course-eval/Bart-Large-CNN')

FOLD = "5"

# TRAIN_PATH = "../Dataset/coursera/coursera_train.txt"
# MODEL_PATH = "facebook/bart-large-cnn"

TRAIN_PATH = "../Dataset/original/org-fold1.txt"
MODEL_PATH = f"outputs/kb-outputs"

with open(TRAIN_PATH, "r",encoding='utf-8') as f:
    file = f.readlines()
train_data = []
for line in file:
    x, y = line.split("\001")[0], line.strip().split("\001")[1]
    train_data.append([x, y])

train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])
steps = [1]
learing_rates = [4e-5]

print("Training Start")
BEST_ACC = 0
for lr in learing_rates:
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
        model = Seq2SeqModel(
            encoder_decoder_type="bart",
            encoder_decoder_name=MODEL_PATH,
            args=model_args,
        )
        BEST_ACC = model.train_model(train_df, BEST_ACC)
print(f"Model Path: {MODEL_PATH}")
print(f"Train dataset: {TRAIN_PATH}")
