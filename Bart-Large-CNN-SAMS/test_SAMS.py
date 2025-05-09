"""
This module is used to test the SAMS (Single-aspect Multi-sentiment)
"""
from simpletransformers.seq2seq import Seq2SeqModel
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig, GPT2Tokenizer, GPT2Model, MBart50TokenizerFast, MBartForConditionalGeneration, GPT2LMHeadModel
import torch
import numpy as np
import os

# PATH = f"../Dataset/coursera/testformat.txt"
PATH = "../Dataset/testset/testformat1.txt"

def predict_val(model, device):
    """
    This function is used to validate the result of the model
    """
    candidate_list = ["positive", "neutral", "negative"]
    model.eval()
    model.config.use_cache = False
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    with open(PATH, "r", encoding="utf-8") as f:
        file = f.readlines()
    count = 0
    total = 0
    true_labels = []
    predicted_labels = []
    for line in file:
        total += 1
        score_list = []
        line = line.strip()
        x, golden_polarity = line.split("\001")[0], line.split("\001")[1]

        input_ids = tokenizer(
            [x] * 3, return_tensors='pt', padding=True, truncation=True, max_length=512
            )['input_ids'].to(device)
        target_list = ["The sentiment of this sentence is" + candi.lower() + " ." for candi in
                       candidate_list]
        output_ids = tokenizer(target_list, return_tensors='pt',
                               padding=True, truncation=True)['input_ids'].to(device)
        model.to(device)
        output_ids = output_ids.to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, decoder_input_ids=output_ids)[0]
            logits = output.softmax(dim=-1).to('cpu').numpy()
        for i in range(3):
            score = 1
            for j in range(logits[i].shape[0] - 2):
                score *= logits[i][j][output_ids[i][j + 1]]
            score_list.append(score)
        predict = candidate_list[np.argmax(score_list)]
        predicted_sentence = target_list[np.argmax(score_list)]
        true_labels.append(golden_polarity)
        predicted_labels.append(predict)
        if predict == golden_polarity:
            count += 1
            print(line, predicted_sentence, " actual:", golden_polarity,
                  "acc:", count/total, count, total)
        else:
            print("Mismatch:",line," ", predicted_sentence, " actual:",
                  golden_polarity,"acc:", count/total, count, total)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels,
                                predicted_labels, average='macro', zero_division=1.0)
    accuracy = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    print(f"Acc: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    print("confusion metrix:", cm)
    print("classification report:",classification_report(true_labels,
                                                         predicted_labels, zero_division=1.0))
    print(f"Current test format text: {PATH}")
    return accuracy, precision, recall, f1

def predict_test(model, device):
    """
    Return the testing result of the model
    """
    candidate_list = ["positive", "neutral", "negative"]
    model.eval()
    model.config.use_cache = False
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    with open(PATH, "r", encoding="utf-8") as f:
        file = f.readlines()
    count = 0
    total = 0
    true_labels = []
    predicted_labels = []
    for line in file:
        total += 1
        score_list = []
        line = line.strip()
        x, golden_polarity = line.split("\001")[0], line.split("\001")[1]
        input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids'].to(device)
        target_list = ["The sentiment of this sentence is " + candi.lower() + " ."
                       for candi in candidate_list]
        output_ids = tokenizer(target_list, return_tensors='pt', padding=True,
                               truncation=True)['input_ids'].to(device)
        model.to(device)
        output_ids = output_ids.to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, decoder_input_ids=output_ids)[0]
            logits = output.softmax(dim=-1).to('cpu').numpy()
        print(logits)
        for i in range(3):
            score = 1
            for j in range(logits[i].shape[0] - 2):
                score *= logits[i][j][output_ids[i][j + 1]]
            score_list.append(score)
        print(score_list)
        predict = candidate_list[np.argmax(score_list)]
        predicted_sentence = target_list[np.argmax(score_list)]
        true_labels.append(golden_polarity)
        predicted_labels.append(predict)
        if predict == golden_polarity:
            count += 1
            print(line, predicted_sentence, " actual:",
                  golden_polarity, "acc:", count/total, count, total)
        else:
            print("Mismatch:",line," ", predicted_sentence, " actual:",
                  golden_polarity,"acc:", count/total, count, total)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='macro', zero_division=1.0)
    accuracy = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}, Acc: {accuracy}")
    print("confusion metrix:", cm)
    print("classification report:",classification_report(true_labels,predicted_labels,
                                                         zero_division=1.0))
    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}, Acc: {accuracy}")

    return accuracy, precision, recall, f1

print("imported test.py successfully")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model2 = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
# acc = predict_test(model2, device)
