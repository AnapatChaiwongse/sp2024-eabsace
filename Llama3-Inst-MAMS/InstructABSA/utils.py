import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Trainer,
    Seq2SeqTrainer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    LlamaTokenizer,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    DefaultDataCollator,
)
from peft import LoraConfig, get_peft_model, TaskType

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # adjust for your model
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

class T5Generator:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        model_inputs = self.tokenizer(sample['text'], max_length=512, truncation=True)
        labels = self.tokenizer(sample["labels"], max_length=64, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """
        #Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
        )

        # Define trainer object
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"] if tokenized_datasets.get("validation") is not None else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(self, tokenized_dataset, batch_size = 4, max_length = 128, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print('Model loaded to: ', self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch, max_length = max_length)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
        return predicted_output
    
    def get_metrics(self, y_true, y_pred, is_triplet_extraction=False):
        total_pred = 0
        total_gt = 0
        tp = 0
        if not is_triplet_extraction:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(', ')
                pred_list = pred.split(', ')
                total_pred+=len(pred_list)
                total_gt+=len(gt_list)
                for gt_val in gt_list:
                    for pred_val in pred_list:
                        if pred_val in gt_val or gt_val in pred_val:
                            tp+=1
                            break

        else:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(', ')
                pred_list = pred.split(', ')
                total_pred+=len(pred_list)
                total_gt+=len(gt_list)
                for gt_val in gt_list:
                    gt_asp = gt_val.split(':')[0]

                    try:
                        gt_op = gt_val.split(':')[1]
                    except:
                        continue

                    try:
                        gt_sent = gt_val.split(':')[2]
                    except:
                        continue

                    for pred_val in pred_list:
                        pr_asp = pred_val.split(':')[0]

                        try:
                            pr_op = pred_val.split(':')[1]
                        except:
                            continue

                        try:
                            pr_sent = gt_val.split(':')[2]
                        except:
                            continue

                        if pr_asp in gt_asp and pr_op in gt_op and gt_sent == pr_sent:
                            tp+=1

        p = tp/total_pred
        r = tp/total_gt
        return p, r, 2*p*r/(p+r), None


class T5Classifier:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download = False)

        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, force_download = True)
        self.model = AutoModelForCausalLM.from_pretrained(model_checkpoint, force_download = False)
        self.model = get_peft_model(self.model, lora_config)

        # self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model, padding=True)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        # self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        sample['input_ids'] = self.tokenizer(sample["text"], max_length = 512, truncation = True, padding='max_length').input_ids
        sample['labels'] = self.tokenizer(sample["labels"], max_length = 512, truncation = True, padding='max_length').input_ids
        # sample['labels'] = sample['input_ids'].copy()
        
        print(f"Input shape: {len(sample['input_ids'])}")
        print(f"Labels shape: {len(sample['labels'])}")
        return sample
    
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """

        # Set training arguments
        # args = Seq2SeqTrainingArguments(
        #     **kwargs
        #     )
        kwargs.pop("predict_with_generate", None)
        kwargs.pop("evaluation_strategy", None)
        args = TrainingArguments(
            **kwargs,
            fp16=True,
            # auto_find_batch_size=True,
            )
        print("per_device_train_batch_size: " + str(args.per_device_train_batch_size))
        print("learning_rate: " + str(args.learning_rate))
        print("fp16: " + str(args.fp16) + "\n")
        print(f'Memory Allocated: {torch.cuda.memory_allocated()} bytes')
        print(f'Memory Cached: {torch.cuda.memory_reserved()} bytes')

        # print(tokenized_datasets["train"][0])
        # Define trainer object
    
        # def debug_collator(batch):
        #     print("batch: " + str(batch))
        #     return data_collator(batch)

        trainer = Trainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"] if tokenized_datasets.get("validation") is not None else None,
            tokenizer=self.tokenizer, 
            data_collator = self.data_collator
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....\n' + "*"*30)
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(self, tokenized_dataset, batch_size = 8, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids

        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        # self.model.to(self.device)
        # print(f'Model {self.model} loaded to: ', self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            batch_texts = self.tokenizer.batch_decode(batch, skip_special_tokens=True)
            batch_texts = batch_texts[0]
            input_ids = self.tokenizer(batch_texts, return_tensors="pt", max_length=1024).input_ids.to(self.device)

            output_ids = self.model.generate(input_ids, num_beams=1, do_sample=True, max_new_tokens=1)
            # output_ids = self.model.generate(
            #     batch,
            #     max_new_tokens=32,
            #     do_sample=True,
            # )
            # print(batch_texts)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            # for output_text in output_texts:
            #     predicted_output.append(output_text)

            predicted_output.append(output_texts[-1].split("\noutput:")[1].strip())
            print("output: " + output_texts[0])
        print(f"predicted_output sample: {predicted_output[0]}")
        return predicted_output

    def get_metrics(self, y_true, y_pred):
        """
        Compute both metrics and roc_auc_score for evaluation mode.
        Need to add auc from sklearn.metrics.roc_auc_score()
        """
        print("Called get_metrics():")
        print(f"y_true sample: {y_true[0]}")
        print(f"y_pred sample: {y_pred[0]}")

        return precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro'), \
            f1_score(y_true, y_pred, average='macro'), accuracy_score(y_true, y_pred)
