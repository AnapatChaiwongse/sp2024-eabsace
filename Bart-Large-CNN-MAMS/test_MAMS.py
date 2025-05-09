from simpletransformers.seq2seq import Seq2SeqModel
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig, GPT2Tokenizer, GPT2Model, MBart50TokenizerFast, MBartForConditionalGeneration, GPT2LMHeadModel
# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)
import torch
import numpy as np
import os

fold = "5"
path = f"../Dataset/testset/testformat{fold}.txt"
# path = "../Dataset/testset/onelinetest.txt"

#def calculate_metrics(predicted, golden):
#    label_mapping = {"positive": 1, "neutral": 2, "negative": 0}

    # Map labels to numeric values and handle 'nan' values
#    golden_numeric = np.array([label_mapping.get(label, -1) for label in golden])

    # Filter out entries with -1 (nan) before further calculations
#    valid_entries = golden_numeric != -1
#    predicted_numeric = np.array([label_mapping[label] for label in predicted])[valid_entries]
#    golden_numeric = golden_numeric[valid_entries]

#    TP = np.sum(np.equal(predicted_numeric, golden_numeric) & np.equal(predicted_numeric, 1))
#    FP = np.sum(np.not_equal(predicted_numeric, golden_numeric) & np.equal(predicted_numeric, 1))
#    FN = np.sum(np.not_equal(predicted_numeric, golden_numeric) & np.equal(predicted_numeric, 0))
#    TN = np.sum(np.equal(predicted_numeric, golden_numeric) & np.equal(predicted_numeric, 0))

#    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
#    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#    return precision, recall, f1_score

def predict_val(model, device):#, output_dir):
    candidate_list = ["positive", "neutral", "negative"]

    #model = MBartForConditionalGeneration.from_pretrained(output_dir)
    #model = MBartForConditionalGeneration.from_pretrained('/ICTeval/ICTeval_code/outputs/mbart')
    model.eval()
    model.config.use_cache = False
    #tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    #tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/mGPT")
    with open(path, "r") as f:
        file = f.readlines()
    train_data = []
    count = 0
    total = 0
    true_labels = []
    predicted_labels = []
    losses =[]
    for line in file:
        total += 1
        # score_list = []
        score_list1 = []
        score_list2 = []
        score_list3 = []
        score_list4 = []
        score_list5 = []
        line = line.strip()
        x, term, golden_polarity = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
        input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids'].to(device)

        # target_list = ["For " + term.lower() + ", the sentiment is " + candi.lower() + " ." for candi in candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list1.append(score)

        target_list = ["The sentiment polarity of " + term.lower() + " is " + candi.lower() + " ." for candi in
                       candidate_list]
        ## target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        ## input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']

        output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
        model.to(device)
        output_ids = output_ids.to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, decoder_input_ids=output_ids)[0]
            logits = output.softmax(dim=-1).to('cpu').numpy()

        #with torch.no_grad():
        #    output = model(input_ids=input_ids.to(device))[0]
        #    logits = output.softmax(dim=-1).to('cpu').numpy()
        for i in range(3):
            score = 1
            for j in range(logits[i].shape[0] - 2):
                #print(logits[i][j][output_ids[i][j + 1]])
                score *= logits[i][j][output_ids[i][j + 1]]
            score_list2.append(score)
        score_list = score_list2

        #loss = -torch.log(torch.tensor(score_list)).mean().item()  # Calculate loss for the example
        #losses.append(loss)

        # target_list = ["The " + term.lower() + " category has a " + candi.lower() + " label ." for candi in
        #                candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list3.append(score)

        # target_list = ["The sentiment is " + candi.lower() + " for " + term.lower() + " ." for candi in
        #                candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list4.append(score)

        # target_list = ["The " + term.lower() + " is " + candi.lower() + " ." for candi in
        #                candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list5.append(score)

        # score_list = [(score_list1[i] + score_list2[i] + score_list3[i]) for i in range(0, len(score_list1))]
        predict = candidate_list[np.argmax(score_list)]
        predicted_sentence = target_list[np.argmax(score_list)]
        true_labels.append(golden_polarity)
        predicted_labels.append(predict)
        #predicted_term = term.lower()
        if predict == golden_polarity:
            count += 1
            print(line, predicted_sentence, " actual:", golden_polarity, "acc:", count/total, count, total)
        else:
            print("Mismatch:",line," ", predicted_sentence, " actual:", golden_polarity,"acc:", count/total, count, total)

    #matrix = self.compute_metrics(np.array(true_labels), np.array(predicted_labels), calculate_metrics=calculate_metrics)
    #precision, recall, f1_score = matrix['calculate_metrics']
    #precision, recall, f1_score = calculate_metrics(np.array(predicted_labels), np.array(true_labels))
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro', zero_division=1.0)
    accuracy = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    print(f"Acc: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    print("confusion metrix:", cm)
    print("classification report:",classification_report(true_labels,predicted_labels, zero_division=1.0))
    print(f"Current test fold: {fold}, Current test format text: {path}")
    return accuracy, precision, recall, f1

def predict_test(model, device):#, output_dir):
    candidate_list = ["positive", "neutral", "negative"]
    #model = MBartForConditionalGeneration.from_pretrained(output_dir)
    #model = MBartForConditionalGeneration.from_pretrained('/ICTeval/ICTeval_code/outputs/checkpoint-5865-epoch-15/mbart')
    model.eval()
    model.config.use_cache = False
    #tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    with open(path, "r") as f:
        file = f.readlines()
    train_data = []
    count = 0
    total = 0
    true_labels = []
    predicted_labels = []
    for line in file:
        total += 1
        # score_list = [] 
        score_list1 = []
        score_list2 = []
        score_list3 = []
        score_list4 = []
        score_list5 = []
        line = line.strip()
        #print(line)
        x, term, golden_polarity = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
        input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids'].to(device)

        # target_list = ["For " + term.lower() + ", the sentiment is " + candi.lower() + " ." for candi in candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list1.append(score)

        target_list = ["The sentiment polarity of " + term.lower() + " is " + candi.lower() + " ." for candi in
                       candidate_list]
        ## target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        ## input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']

        output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
        model.to(device)
        output_ids = output_ids.to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, decoder_input_ids=output_ids)[0]
            logits = output.softmax(dim=-1).to('cpu').numpy()
        print(logits)
        #with torch.no_grad():
        #    output = model(input_ids=input_ids.to(device))[0]
        #    logits = output.softmax(dim=-1).to('cpu').numpy()
        for i in range(3):
            score = 1
            # print(logits[i].shape[0])
            for j in range(logits[i].shape[0] - 2):
                #print(logits[i][j][output_ids[i][j + 1]])
                # print(f"logits[{str(i)}][{str(j)}][{output_ids[i][j+1]}]: ", logits[i][j][output_ids[i][j + 1]])
                score *= logits[i][j][output_ids[i][j + 1]]
            score_list2.append(score)
        score_list = score_list2

        # target_list = ["The " + term.lower() + " category has a " + candi.lower() + " label ." for candi in
        #                candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list3.append(score)

        # target_list = ["The sentiment is " + candi.lower() + " for " + term.lower() + " ." for candi in
        #                candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list4.append(score)

        # target_list = ["The " + term.lower() + " is " + candi.lower() + " ." for candi in
        #                candidate_list]
        # # target_list = ["For " + term.lower() + ", it is a " + candi.lower() + "sentence ." for candi in candidate_list]
        # # input_ids = tokenizer([x] * 3, return_tensors='pt')['input_ids']
        # output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        # with torch.no_grad():
        #     output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        #     logits = output.softmax(dim=-1).to('cpu').numpy()
        # for i in range(3):
        #     score = 1
        #     for j in range(logits[i].shape[0] - 2):
        #         score *= logits[i][j][output_ids[i][j + 1]]
        #     score_list5.append(score)

        # score_list = [(score_list1[i] + score_list2[i] + score_list3[i]) for i in range(0, len(score_list1))]
        print(score_list)
        predict = candidate_list[np.argmax(score_list)]
        predicted_sentence = target_list[np.argmax(score_list)]
        true_labels.append(golden_polarity)
        predicted_labels.append(predict)
        # with open(f'course-eval/Dataset/Original/Splited/1-fold/Predict/CNN_Original_test{fold}_predict.txt', 'a') as f0:
        #            f0.writelines(predicted_sentence + "\n")
        # with open(f'course-eval/Dataset/Original/Splited/1-fold/Predict/CNN_Original_true{fold}_predict.txt', 'a') as f0:
        #            f0.writelines("The sentiment polarity of " + term.lower() + " is " + golden_polarity + " ." +"\n")
        predicted_term = term.lower()
        if predict == golden_polarity:
            count += 1
            print(line, predicted_sentence, " actual:", golden_polarity, "acc:", count/total, count, total)
        else:
            print("Mismatch:",line," ", predicted_sentence, " actual:", golden_polarity,"acc:", count/total, count, total) 

    #matrix = self.compute_metrics(np.array(true_labels), np.array(predicted_labels), calculate_metrics=calculate_metrics)
    #precision, recall, f1_score = matrix['calculate_metrics']
    #precision, recall, f1_score = calculate_metrics(np.array(predicted_labels), np.array(true_labels))
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro', zero_division=1.0)
    accuracy = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}, Acc: {accuracy}")
    print("confusion metrix:", cm)
    print("classification report:",classification_report(true_labels,predicted_labels, zero_division=1.0))
    #print(matrix)
    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}, Acc: {accuracy}")

    return accuracy, precision, recall, f1

print("imported test.py successfully")
# print(f"Current path: {os.getcwd()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model2 = BartForConditionalGeneration.from_pretrained(f'outputs/Original/{int(fold)-1}-fold').to(device)
# print(f"Model: {model2.model}, Device: {device}")
# model2 = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
# acc = predict_test(model2, device)

