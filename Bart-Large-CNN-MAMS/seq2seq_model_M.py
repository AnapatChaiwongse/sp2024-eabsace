import json
import logging
import math
import os
import random
import warnings
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from test_MAMS import predict_val, predict_test
import matplotlib
import matplotlib.pyplot as plt
font = {'size' : 18}
matplotlib.rc('font', **font)
# import test_Rest14

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers import (
    # AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    BartModel, #add for https://huggingface.co/facebook/bart-base
    pipeline, #add for https://huggingface.co/facebook/bart-large-cnn
    GPT2Tokenizer, #add for https://huggingface.co/gpt2
    GPT2Model, #add for https://huggingface.co/gpt2
    BertConfig,
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    CamembertConfig,
    CamembertModel,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraModel,
    ElectraTokenizer,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    LongformerConfig,
    LongformerModel,
    LongformerTokenizer,
    MarianConfig,
    MarianMTModel,
    MarianTokenizer,
    MobileBertConfig,
    MobileBertModel,
    MobileBertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    MBartForConditionalGeneration, #add for https://huggingface.co/facebook/mbart-large-50
    MBart50TokenizerFast, #add for https://huggingface.co/facebook/mbart-large-50
    get_linear_schedule_with_warmup,
    MBartConfig,
    GPT2LMHeadModel, #https://github.com/ai-forever/mgpt
    GPT2Tokenizer, #https://github.com/ai-forever/mgpt
    GPT2Config,
    get_linear_schedule_with_warmup,
)

from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import Seq2SeqArgs
from simpletransformers.seq2seq.seq2seq_utils import Seq2SeqDataset, SimpleSummarizationDataset

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModel, AutoTokenizer),
    "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),
    "bert": (BertConfig, BertModel, BertTokenizer),
    "camembert": (CamembertConfig, CamembertModel, CamembertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer),
    "electra": (ElectraConfig, ElectraModel, ElectraTokenizer),
    "longformer": (LongformerConfig, LongformerModel, LongformerTokenizer),
    "mobilebert": (MobileBertConfig, MobileBertModel, MobileBertTokenizer),
    "marian": (MarianConfig, MarianMTModel, MarianTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "gpt2": (GPT2Model,GPT2Tokenizer),
    "mbart": (MBartConfig, MBartForConditionalGeneration, MBart50TokenizerFast),
    "mgpt": (GPT2Config, GPT2LMHeadModel ,GPT2Tokenizer)

}



class Seq2SeqModel:
    def __init__(
        self,
        encoder_type=None,
        encoder_name=None,
        decoder_name=None,
        encoder_decoder_type=None,
        encoder_decoder_name=None,
        config=None,
        args=None,
        use_cuda=True,
        cuda_device=0,
        **kwargs,
    ):

        """
        Initializes a Seq2SeqModel.

        Args:
            encoder_type (optional): The type of model to use as the encoder.
            encoder_name (optional): The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            decoder_name (optional): The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
                                    Must be the same "size" as the encoder model (base/base, large/large, etc.)
            encoder_decoder_type (optional): The type of encoder-decoder model. (E.g. bart)
            encoder_decoder_name (optional): The path to a directory containing the saved encoder and decoder of a Seq2SeqModel. (E.g. "outputs/") OR a valid BART or MarianMT model.
            config (optional): A configuration file to build an EncoderDecoderModel.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        if not config:
            # if not ((encoder_name and decoder_name) or encoder_decoder_name) and not encoder_type:
            if not ((encoder_name and decoder_name) or encoder_decoder_name):
                raise ValueError(
                    "You must specify a Seq2Seq config \t OR \t"
                    "encoder_type, encoder_name, and decoder_name OR \t \t"
                    "encoder_type and encoder_decoder_name"
                )
            elif not (encoder_type or encoder_decoder_type):
                raise ValueError(
                    "You must specify a Seq2Seq config \t OR \t"
                    "encoder_type, encoder_name, and decoder_name \t OR \t"
                    "encoder_type and encoder_decoder_name"
                )

        self.args = self._load_model_args(encoder_decoder_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, Seq2SeqArgs):
            self.args = args

        if "sweep_config" in kwargs:
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = {key: value["value"] for key, value in sweep_config.as_dict().items() if key != "_wandb"}
            self.args.update_from_dict(sweep_values)

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if use_cuda:
            if torch.cuda.is_available():
                print("cuda is used")
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            print("cpu is used")
            self.device = "cpu"

        self.results = {}

        self.encoder_tokenizer = None
        self.decoder_tokenizer = None

        if not use_cuda:
            self.args.fp16 = False

        # config = EncoderDecoderConfig.from_encoder_decoder_configs(config, config)
        if encoder_decoder_type:
            config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_decoder_type]
        else:
            config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]
        print(f"Initialize the model and token")
        # AutoTokenizer tokenizer_class
        # AutoModelForSeq2SeqLM model_class

        if encoder_decoder_type in ["bart", "marian"]:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(encoder_decoder_name)
            print(f"Model: {self.model}")
            if encoder_decoder_type == "bart":
                self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_decoder_name)
                print(f"Token: {self.encoder_tokenizer}")
            elif encoder_decoder_type == "marian":
                if self.args.base_marian_model_name:
                    self.encoder_tokenizer = AutoTokenizer.from_pretrained(self.args.base_marian_model_name)
                else:
                    self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_decoder_name)
                print(f"Token: {self.encoder_tokenizer}")
            self.decoder_tokenizer = self.encoder_tokenizer
            self.config = self.model.config
        elif encoder_decoder_type == "mbart":
            self.model = MBartForConditionalGeneration.from_pretrained(encoder_decoder_name)
            self.encoder_tokenizer = MBart50TokenizerFast.from_pretrained(encoder_decoder_name)
            self.decoder_tokenizer = self.encoder_tokenizer  
        #elif encoder_decoder_type == "mgpt":
        #        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        #            os.path.join(encoder_decoder_name, "encoder"), os.path.join(encoder_decoder_name, "decoder")
        #        )    
        #        self.model = GPT2LMHeadModel.from_pretrained(self.model)
        #        self.decoder_tokenizer = GPT2Tokenizer.from_pretrained(self.model)
        #        self.encoder_tokenizer = self.decoder_tokenizer
        else:
            if encoder_decoder_name:
                # self.model = EncoderDecoderModel.from_pretrained(encoder_decoder_name)
                self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                    os.path.join(encoder_decoder_name, "encoder"), os.path.join(encoder_decoder_name, "decoder")
                )
                self.model.encoder = model_class.from_pretrained(os.path.join(encoder_decoder_name, "encoder"))
                self.model.decoder = BertForMaskedLM.from_pretrained(os.path.join(encoder_decoder_name, "decoder"))
                self.encoder_tokenizer = tokenizer_class.from_pretrained(os.path.join(encoder_decoder_name, "encoder"))
                self.decoder_tokenizer = BertTokenizer.from_pretrained(os.path.join(encoder_decoder_name, "decoder"))
            else:
                self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                    encoder_name, decoder_name, config=config
                )
                self.encoder_tokenizer = tokenizer_class.from_pretrained(encoder_name)
                self.decoder_tokenizer = BertTokenizer.from_pretrained(decoder_name)
            self.encoder_config = self.model.config.encoder
            self.decoder_config = self.model.config.decoder

        #print("encoder_tokenizer:", self.encoder_tokenizer)
        #print("decoder_tokenizer:", self.decoder_tokenizer)    

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None

        if encoder_decoder_name:
            self.args.model_name = encoder_decoder_name

            # # Checking if we are loading from a saved model or using a pre-trained model
            # if not saved_model_args and encoder_decoder_type == "marian":
            # Need to store base pre-trained model name to get the tokenizer when loading a saved model
            self.args.base_marian_model_name = encoder_decoder_name

        elif encoder_name and decoder_name:
            self.args.model_name = encoder_name + "-" + decoder_name
        else:
            self.args.model_name = "encoder-decoder"

        if encoder_decoder_type:
            self.args.model_type = encoder_decoder_type
        elif encoder_type:
            self.args.model_type = encoder_type + "-bert"
        else:
            self.args.model_type = "encoder-decoder"

    def train_model(
        self, train_data, best_accuracy, output_dir=None, show_running_loss=True, args=None, eval_data=None, verbose=True, **kwargs,
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down training significantly as the predicted sequences need to be generated.

        Returns:
            None
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        # if self.args.silent:
        #     show_running_loss = False

        if self.args.evaluate_during_training and eval_data is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_data is not specified."
                " Pass eval_data to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        if self.encoder_tokenizer is None or self.decoder_tokenizer is None:
            raise ValueError("Tokenizer initialization failed.")

        train_dataset = self.load_and_cache_examples(train_data, verbose=verbose)

        os.makedirs(output_dir, exist_ok=True)

        global_step, tr_loss, best_accuracy = self.train(
            train_dataset,
            output_dir,
            best_accuracy,
            show_running_loss=show_running_loss,
            eval_data=eval_data,
            verbose=verbose,
            **kwargs,
        )

        self._save_model(self.args.output_dir, model=self.model)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        self.encoder_tokenizer.save_pretrained(output_dir)
        self.decoder_tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_name, output_dir))

        return best_accuracy

    def train(
        self, train_dataset, output_dir, best_accuracy, show_running_loss=True, eval_data=None, verbose=True, **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args

        tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in model.named_parameters() if n in params]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        # TODO: Use custom optimizer like with BertSum?
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        if (
            args.model_name
            and os.path.isfile(os.path.join(args.model_name, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name, "scheduler.pt")))

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info(" Training started")

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0)
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info("   Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)

        if args.wandb_project:
            wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)
            wandb.watch(self.model)

        if args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        hist_graph = {
        'train_loss': [],
        'val_loss': [],
         }
        predicted_values = []
        real_values = []
        model.train()
        #eps = 1e-6
        torch.autograd.set_detect_anomaly(True)
        for current_epoch in train_iterator:
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                # batch = tuple(t.to(device) for t in batch)

                inputs = self._get_inputs_dict(batch)        
               # if any(torch.isnan(value).any() or torch.isinf(value).any() for value in inputs.values()):
                #    print("NaN or Inf found in input tensor.")
                #    print("Inputs:", inputs)
                #    break  # Optional: Stop training after the first occurrence
                #print("Inputs:", inputs)
                #if args.fp16:
                #    with amp.autocast():
                #        outputs = model(**inputs)
                    # Check for NaN in the model output
                #        if torch.isnan(outputs[0]).any() or torch.isinf(outputs[0]).any():
                #            print("NaN or Inf found in model output.")
                #            print("Outputs:", outputs)
                #            break  # Optional: Stop training after the first occurrence
                #        loss = outputs[0]
                #else:
                #    outputs = model(**inputs)
                    # Check for NaN in the model output
                #    print("Outputs:", outputs)
                #    if torch.isnan(outputs[0]).any() or torch.isinf(outputs[0]).any():
                #        print("NaN or Inf found in model output.")
                #        print("Outputs:", outputs)
                #        break  # Optional: Stop training after the first occurrence
                #    loss = outputs[0]    
                #print(input)
                #print("\n")
                #for key, value in inputs.items():
                #    if torch.is_tensor(value) and (torch.isnan(value).any() or torch.isinf(value).any()):
                #        print(f"Input data for key '{key}' contains NaN or infinite values.")
                #        print("Inputs:", value)
                #        break  # Optional: Stop training after the first occurrence

                #if torch.is_tensor(inputs.get("labels")) and (torch.isnan(inputs["labels"]).any() or torch.isinf(inputs["labels"]).any()):
                #    print("Labels data contains NaN or infinite values.")
                #    print("Labels:", inputs["labels"])
                #    break  # Optional: Stop training after the first occurrence

            #    for key, value in inputs.items():
            #        print(f"Key: {key}, NaN: {torch.isnan(value).any()}, Inf: {torch.isinf(value).any()}\n")
            #    for param in model.parameters():
            #        print(param.grad)
            #    print("Softmax input:", your_softmax_input)
            #    if args.n_gpu > 1:
            #        loss = loss.mean()  # mean() to average on multi-gpu parallel training

                #current_loss = loss.item()
                #print("Loss:", current_loss)
        #        if args.fp16:
        #            with amp.autocast():
        #                print(inputs)
        #                outputs = model(**inputs)
        #                print(outputs)
        #                # model outputs are always tuple in pytorch-transformers (see doc)
        #                loss = outputs[0]
        #        else:
        #            print(inputs)
        #            outputs = model(**inputs)
        #            print(outputs)
        #            # model outputs are always tuple in pytorch-transformers (see doc)
        #            loss = outputs[0]
                #if args.fp16:
                #    with amp.autocast():
                    # Check for NaN or infinite values in input tensors
                #        for key, value in inputs.items():
                #            if torch.isnan(value).any() or torch.isinf(value).any():
                #                print(f"Input tensor '{key}' contains NaN or infinite values.")
                #                break  # Stop processing further if NaN or infinite values are found
                #            else:  # If no NaN or infinite values are found in any input tensor
                #                print("No NaN or infinite values found in input tensors.")
                #    outputs = model(**inputs)
                #    loss = outputs[0]
                #else:
                    # Check for NaN or infinite values in input tensors
                #    for key, value in inputs.items():
                #        if torch.isnan(value).any() or torch.isinf(value).any():
                #            print(f"Input tensor '{key}' contains NaN or infinite values.")
                #            break  # Stop processing further if NaN or infinite values are found
                #        else:  # If no NaN or infinite values are found in any input tensor
                #            print("No NaN or infinite values found in input tensors.")
                
                outputs = model(**inputs)
                #print(outputs)
                loss = outputs[0]    

                logits = outputs.logits
                predicted_token_ids = torch.argmax(logits, dim=-1)

                input_token_ids = inputs["input_ids"]
                input_sentences = []
                for token_ids in input_token_ids:
                    sentence = self.encoder_tokenizer.decode(token_ids, skip_special_tokens=True)
                    input_sentences.append(sentence)

                predicted_sentences = []
                for token_ids in predicted_token_ids:
                    sentence = self.decoder_tokenizer.decode(token_ids, skip_special_tokens=True)
                    predicted_sentences.append(sentence)

                for input_sentence, predicted_sentence in zip(input_sentences, predicted_sentences):
                    print("Input Sentence:", input_sentence)
                    print("Predicted Sentence:", predicted_sentence)
                    print()  # Add a newline for better readability 
                print("\n")    
        #        if args.n_gpu > 1:
        #            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        #        if loss.isnan(): 
        #            current_loss=eps
        #        else:
        #            current_loss = loss.item()
                current_loss = loss.item()
        #        print(loss)
                #print("\n")
                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                #optimizer.step()
                #model.eval()
                #with torch.no_grad():
                #    input_ids = inputs["input_ids"]
                #   output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)[0]
                #    logits = output.softmax(dim=-1).to('cpu').numpy()

                #    predicted_values.extend(logits)
                #    real_values.extend(inputs["labels"])
                #model.train()
                #print("Loss:", loss.item())
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()   
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1                
                


                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss
                        if args.wandb_project:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self._save_model(output_dir_current, optimizer, scheduler, model=model)

                    if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = self.eval_model(
                            eval_data,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            **kwargs,
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        if args.save_eval_checkpoints:
                            self._save_model(output_dir_current, optimizer, scheduler, model=model, results=results)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
                        )

                        if args.wandb_project:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            if args.save_best_model:
                                self._save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results
                                )
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                if args.save_best_model:
                                    self._save_model(
                                        args.best_model_dir, optimizer, scheduler, model=model, results=results
                                    )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return global_step, tr_loss / global_step
                        else:
                            if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                if args.save_best_model:
                                    self._save_model(
                                        args.best_model_dir, optimizer, scheduler, model=model, results=results
                                    )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return global_step, tr_loss / global_step
                                        

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            #train_accuracy = accuracy_score(real_values, predicted_values)
            #train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(real_values, predicted_values, average='macro', zero_division=1.0)
            val_accuracy,val_precision, val_recall, val_f1_score = predict_val(model, self.device)#, output_dir_current)
            print('batch: '+str(args.train_batch_size)+' accumulation_steps: '+str(args.gradient_accumulation_steps)+\
                ' lr: '+str(args.learning_rate)+' epochs: '+str(args.num_train_epochs)+' epoch: '+str(epoch_number))
            #if val_accuracy > best_accuracy:
            #    best_accuracy = val_accuracy
            #    print('---test dataset----')
            #    test_acc,test_precision, test_recall, test_f1_score = predict_test(model, self.device)#, output_dir_current)
            #    with open('/ICTeval/ICTeval_single/courseEval250/output.txt', 'a') as f0:
            #        f0.writelines('batch: '+str(args.train_batch_size)+' accumulation_steps: '+str(args.gradient_accumulation_steps)+\
            #                      ' lr: '+str(args.learning_rate)+' epochs: '+str(args.num_train_epochs)+' epoch: '+str(epoch_number)+' val_accuracy: '+str(best_accuracy)+\
            #                      ' val_precision: ' +str(val_precision) +' val_recall: '  +str(val_recall) +' val_f1_score: ' +str(val_f1_score) +' test_accuracy: '+str(test_acc)+\
            #                      ' test_precision: ' +str(test_precision) +' test_recall: ' +str(test_recall) +' test_f1_score: ' +str(test_f1_score)+'\n')
            #hist_graph['train_loss'].append(tr_loss)
            #hist_graph['train_acc'].append(train_accuracy)
            #hist_graph['train_precision'].append(train_precision)
            #hist_graph['train_recall'].append(train_recall)
            #hist_graph['train_f1'].append(train_f1)
            #hist_graph['val_loss'].append(val_avg_loss)

            #fig, ax = plt.subplots(figsize=(8,6))
            #ax.plot(hist_graph['train_loss'], label='train_loss')
            #ax.plot(hist_graph['val_loss'], label='val_loss')
            #ax.set_ylabel('Loss')
            #ax.set_xlabel('Epochs')
            #plt.legend()
            #plt.savefig('graph_loss.png')

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self._save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training:
                results = self.eval_model(
                    eval_data,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    **kwargs,
                )

                if args.save_eval_checkpoints:
                    self._save_model(output_dir_current, optimizer, scheduler, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args.output_dir, "training_progress_scores.csv"), index=False)

                if args.wandb_project:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    if args.save_best_model:
                        self._save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        if args.save_best_model:
                            self._save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return global_step, tr_loss / global_step
                else:
                    if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        if args.save_best_model:
                            self._save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return global_step, tr_loss / global_step

        return global_step, tr_loss / global_step, best_accuracy

    def eval_model(self, eval_data, output_dir=None, verbose=True, silent=False, **kwargs):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.
        Returns:
            results: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        eval_dataset = self.load_and_cache_examples(eval_data, evaluate=True, verbose=verbose, silent=silent)
        os.makedirs(output_dir, exist_ok=True)

        result = self.evaluate(eval_dataset, output_dir, verbose=verbose, silent=silent, **kwargs)
        self.results.update(result)

        if self.args.evaluate_generated_text:
            to_predict = eval_data["input_text"].tolist()
            preds = self.predict(to_predict)

            result = self.compute_metrics(eval_data["target_text"].tolist(), preds, **kwargs)
            self.results.update(result)

        if verbose:
            logger.info(self.results)

        return self.results

    def evaluate(self, eval_dataset, output_dir, verbose=True, silent=False, **kwargs):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        eval_output_dir = output_dir

        results = {}

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        for batch in tqdm(eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"):
            # batch = tuple(t.to(device) for t in batch)

            inputs = self._get_inputs_dict(batch)
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs[0]
                eval_loss += loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        results["eval_loss"] = eval_loss

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

        return results

    def predict(self, to_predict):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction. Note that the prefix should be prepended to the text.

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        self._move_model_to_device()

        all_outputs = []
        # Batching
        for batch in [
            to_predict[i : i + self.args.eval_batch_size] for i in range(0, len(to_predict), self.args.eval_batch_size)
        ]:
            if self.args.model_type == "marian":
                input_ids = self.encoder_tokenizer.prepare_translation_batch(
                    batch, max_length=self.args.max_seq_length, pad_to_max_length=True, return_tensors="pt",
                )["input_ids"]
            else:
                input_ids = self.encoder_tokenizer.batch_encode_plus(
                    batch, max_length=self.args.max_seq_length, pad_to_max_length=True, return_tensors="pt",
                )["input_ids"]
            input_ids = input_ids.to(self.device)

            if self.args.model_type in ["bart", "marian","mbart"]:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                )
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    decoder_start_token_id=self.model.config.decoder.pad_token_id,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                )

            all_outputs.extend(outputs.cpu().numpy())

        if self.args.use_multiprocessed_decoding:
            self.model.to("cpu")
            with Pool(self.args.process_count) as p:
                outputs = list(
                    tqdm(
                        p.imap(self._decode, all_outputs, chunksize=self.args.multiprocessing_chunksize),
                        total=len(all_outputs),
                        desc="Decoding outputs",
                        disable=self.args.silent,
                    )
                )
            self._move_model_to_device()
        else:
            outputs = [
                self.decoder_tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for output_id in all_outputs
            ]

        if self.args.num_return_sequences > 1:
            return [
                outputs[i : i + self.args.num_return_sequences]
                for i in range(0, len(outputs), self.args.num_return_sequences)
            ]
        else:
            return outputs

    def _decode(self, output_id):
        return self.decoder_tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def compute_metrics(self, labels, preds, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            labels: List of target sequences
            preds: List of model generated outputs
            **kwargs: Custom metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.

        Returns:
            result: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"
        # assert len(labels) == len(preds)

        results = {}
        for metric, func in kwargs.items():
            results[metric] = func(labels, preds)

        return results

    def load_and_cache_examples(self, data, evaluate=False, no_cache=False, verbose=True, silent=False):
        """
        Creates a T5Dataset from data.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        encoder_tokenizer = self.encoder_tokenizer
        decoder_tokenizer = self.decoder_tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        if args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(encoder_tokenizer, decoder_tokenizer, args, data, mode)
        else:
            if args.model_type in ["bart", "marian"]:
                return SimpleSummarizationDataset(encoder_tokenizer, self.args, data, mode)
            else:
                return Seq2SeqDataset(encoder_tokenizer, decoder_tokenizer, self.args, data, mode,)

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "eval_loss": [],
            "train_loss": [],
            **extra_metrics,
        }

        return training_progress_scores

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving model into {output_dir}")

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            self._save_model_args(output_dir)

            if self.args.model_type in ["bart", "marian"]:
                os.makedirs(os.path.join(output_dir), exist_ok=True)
                model_to_save.save_pretrained(output_dir)
                self.config.save_pretrained(output_dir)
                if self.args.model_type == "bart":
                    self.encoder_tokenizer.save_pretrained(output_dir)

            elif self.args.model_type == "mbart":
                os.makedirs(os.path.join(output_dir, "mbart"), exist_ok=True)
                model_to_save.save_pretrained(os.path.join(output_dir, "mbart"))
                self.encoder_tokenizer.save_pretrained(os.path.join(output_dir, "mbart"))
            else:
                os.makedirs(os.path.join(output_dir, "encoder"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "decoder"), exist_ok=True)

                self.encoder_config.save_pretrained(os.path.join(output_dir, "encoder"))
                self.decoder_config.save_pretrained(os.path.join(output_dir, "decoder"))

                model_to_save = (
                    self.model.encoder.module if hasattr(self.model.encoder, "module") else self.model.encoder
                )
                model_to_save.save_pretrained(os.path.join(output_dir, "encoder"))

                model_to_save = (
                    self.model.decoder.module if hasattr(self.model.decoder, "module") else self.model.decoder
                )

                model_to_save.save_pretrained(os.path.join(output_dir, "decoder"))

                self.encoder_tokenizer.save_pretrained(os.path.join(output_dir, "encoder"))
                self.decoder_tokenizer.save_pretrained(os.path.join(output_dir, "decoder"))

            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_inputs_dict(self, batch):
        print(batch)
        device = self.device
        if self.args.model_type in ["bart", "marian"]:
            pad_token_id = self.decoder_tokenizer.pad_token_id
            source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
            y_ids = y[:, :-1].contiguous()
            labels = y[:, 1:].clone()
            labels[y[:, 1:] == pad_token_id] = -100

            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "labels": labels.to(device),
            }
        else:
            labels = batch[1]
            labels_masked = labels.clone()
            labels_masked[labels_masked == self.decoder_tokenizer.pad_token_id] = -100

            inputs = {
                "input_ids": batch[0].to(device),
                "decoder_input_ids": labels.to(device),
                "labels": labels_masked.to(device),
            }

        return inputs

    def _save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = Seq2SeqArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
