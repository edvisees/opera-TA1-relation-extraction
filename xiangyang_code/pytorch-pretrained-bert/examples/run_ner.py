# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForTokenClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from sklearn.metrics import f1_score

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class NERExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask,
        segment_ids, label_id, label_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_mask = label_mask

def read_sent(sent):
    examples = []
    example = ''
    label = ''
    idx = 0
    for i, w in enumerate(sent.words):
        example = ' '.join((example, w.word))
        label = ' '.join((label, 'O'))
    guid = str(idx)
    examples.append(NERExample(guid, example.strip(), label.strip()))
    return examples
num_labels = 17
tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
    do_lower_case=False, do_basic_tokenize=False)    
model_dir = '../ner_bert_finetune'
output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
output_config_file = os.path.join(model_dir, CONFIG_NAME)
config = BertConfig(output_config_file)
model = BertForTokenClassification(config, num_labels=num_labels)
model.load_state_dict(torch.load(output_model_file))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model.to(device)
def pred_ner(sent):
    
    eval_examples = read_sent(sent)
    label_list = get_labels()
    eval_features = convert_examples_to_features(
            eval_examples, label_list, 128, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_label_masks = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask,
        all_segment_ids, all_label_ids, all_label_masks)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    pred_list = []
    for input_ids, input_mask, segment_ids, label_ids, label_masks in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        label_masks = label_masks.to(device)
        with torch.no_grad():
            #tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids, label_masks)
            logits = model(input_ids, segment_ids, input_mask)
        active_loss = label_masks.view(-1) == 1
        active_logits = logits.view(-1, num_labels)[active_loss]
        active_labels = label_ids.view(-1)[active_loss]
        active_logits = active_logits.detach().cpu().numpy()
        active_preds = np.argmax(active_logits, axis=1)
        pred_list.extend(active_preds)
    pred_ner_list = [label_list[p] for p in pred_list]
    sen_len = 0
    for w in sent.words:
        sen_len += 1
    assert len(pred_ner_list) == sen_len
    return pred_ner_list

def read_ner_example(file):
    logger.info("LOOKING AT {}".format(file))
    examples = []
    with open(file, encoding='utf-8') as f:
        example = ''
        label = ''
        idx = 0
        for i, line in enumerate(f):

            if len(line) == 1:
                guid = str(idx)
                idx += 1
                examples.append(
                    NERExample(guid, example.strip(), label.strip()))
                example = ''
                label = ''
                continue
            line_split = line.split()
            example = ' '.join((example, line_split[1]))
            label = ' '.join((label, line_split[4]))
            if line_split[4] not in get_labels():
                print(line_split[4])
    return examples

def get_labels():
    return ['B-GPE', 'I-GPE', 'O', 'B-PER', 'I-PER',
    'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-TTL', 'I-TTL',
    'I-FAC', 'B-FAC', 'B-VEH', 'I-VEH', 'B-WEA', 'I-WEA']

def tokenize_label(text, token_text, label):
    text = text.split()
    label = label.split()
    assert len(text) == len(label)
    token_label = []
    idx = 0
    token_text_no = []
    for tt in token_text:
        if not tt.startswith('##'):
            token_text_no.append(tt)
    if len(token_text_no) != len(text):
        print(token_text_no)
        print(text)
        print(label)

    for tt in token_text:
        
        if tt.startswith('##'):
            token_label.append('X')
        else:
            token_label.append(label[idx])
            idx += 1
    assert idx == len(label)
    assert len(token_label) == len(token_text)
    return token_label

def convert_labels_to_idx(labels, label_map):
    labels_ids = []
    label_mask = []
    for l in labels:
        if l == 'X':
            labels_ids.append(-1)
            label_mask.append(0)
        else:
            if l not in label_map:
                logger.info("label %s not in label map" %(l))
            labels_ids.append(label_map[l])
            label_mask.append(1)
    return labels_ids, label_mask

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        #print(example.text_a)
        tokens_a = tokenizer.tokenize(example.text_a)
        labels_a = tokenize_label(example.text_a, tokens_a, example.label)
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            labels_a = labels_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        labels = ['X'] + labels_a + ['X']
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids, label_mask = convert_labels_to_idx(labels, label_map)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_ids += padding
        label_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(label_mask) == max_seq_length

        #label_id = label_map[example.label]
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s " % " ".join(
                    [str(x) for x in labels]))
            logger.info("label_ids: %s" % " ".join(
                    [str(x) for x in label_ids]))
            logger.info("label_mask: %s" % " ".join(
                    [str(x) for x in label_mask]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              label_mask=label_mask))
    return features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file",
                        default='../../data/eng-2015.conll',
                        type=str,
                        required=True,
                        help="train file path")
    parser.add_argument("--dev_file",
                        default='../../data/eng-2016.conll',
                        type=str,
                        required=True,
                        help="dev file path")
    
    parser.add_argument("--bert_model", default='bert-base-cased', type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--finetune_dir",
                        default='NER_BERT',
                        type=str,
                        required=False,
                        help="The output")

    parser.add_argument("--output_dir",
                        default='NER_BERT',
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_finetune",
                        action='store_true',
                        help="Whether to run finetuning.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    
    args = parser.parse_args()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
        do_lower_case=args.do_lower_case, do_basic_tokenize=False)
    label_list = get_labels()
    num_labels = len(label_list)
    train_examples = read_ner_example(args.train_file)
    num_train_optimization_steps = None
    if args.do_train:
        #train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForTokenClassification.from_pretrained(args.bert_model,
              cache_dir=cache_dir,
              num_labels = num_labels)
    if args.fp16:
        model.half()
    if args.do_finetune:
        if not os.path.exists(args.finetune_dir) and not os.listdir(args.finetune_dir):
            raise ValueError("Finetune directory ({}) is empty.".format(args.finetune_dir))
        finetune_model_file = os.path.join(args.finetune_dir, WEIGHTS_NAME)
        finetune_config_file = os.path.join(args.finetune_dir, CONFIG_NAME)
        config = BertConfig(finetune_config_file)
        #model = BertForTokenClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(finetune_model_file))
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_label_masks = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_label_masks)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, label_masks = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids, label_masks)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BertForTokenClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
    else:
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        config = BertConfig(output_config_file)
        model = BertForTokenClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))
        #model = BertForTokenClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_ner_example(args.dev_file)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_label_masks = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask,
            all_segment_ids, all_label_ids, all_label_masks)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        pred_list = []
        label_list = []
        for input_ids, input_mask, segment_ids, label_ids, label_masks in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            label_masks = label_masks.to(device)
            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids, label_masks)
                logits = model(input_ids, segment_ids, input_mask)
            active_loss = label_masks.view(-1) == 1
            active_logits = logits.view(-1, num_labels)[active_loss]
            #print(active_logits.shape)
            active_labels = label_ids.view(-1)[active_loss]
            active_logits = active_logits.detach().cpu().numpy()
            #print(active_logits.shape)
            active_labels = active_labels.to('cpu').numpy()
            active_preds = np.argmax(active_logits, axis=1)
            #print(active_labels.shape, active_preds.shape)
            #tmp_eval_accuracy = accuracy(logits, label_ids, label_masks)

            #eval_loss += tmp_eval_loss.mean().item()
            #eval_accuracy += tmp_eval_accuracy
            pred_list.extend(active_preds)
            label_list.extend(active_labels)
            #print(active_labels.shape)
            nb_eval_examples += active_labels.shape[0]
            nb_eval_steps += 1

        #eval_loss = eval_loss / nb_eval_steps
        #eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss/nb_tr_steps if args.do_train else None
        eval_f1_micro = f1_score(label_list, pred_list, average='micro')
        eval_f1_none = f1_score(label_list, pred_list, average=None)
        result = {'eval_f1_micro': eval_f1_micro,
                  'eval_f1_none': eval_f1_none,
                  'global_step': global_step,
                  'loss': loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key]))) 
        output_pred_file = os.path.join(args.output_dir, "pred_results.conll")
        label_map = get_labels()
        print(len(label_list), len(pred_list))
        with open(output_pred_file, 'w') as f, open(args.dev_file) as dev_f:
            idx = 1
            for l, p, dl in zip(label_list, pred_list, dev_f):
                if len(dl) == 0:
                    print(dl)
                    f.write('\n')
                    idx = 1
                    continue
                f.write(' '.join((str(idx), label_map[l], label_map[p])) + '\n')
                idx += 1




#if __name__ == "__main__":
#    main()






