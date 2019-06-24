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
from pytorch_pretrained_bert.modeling import BERTPCNNNRE, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from sklearn.metrics import f1_score

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class NREExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, en1, en2, label=None):
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
        self.en1 = en1
        self.en2 = en2
        #assert (self.en1 in self.text_a) and (self.en2 in self.text_a)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids,
        lpos, rpos, label_id, tokens_len, pcnn_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.lpos = lpos
        self.rpos = rpos
        self.len = tokens_len
        self.pcnn_mask = pcnn_mask
        #self.label_mask = label_mask

# Just for prediction
def read_sent(sents):
    examples = []
    
    for i, sent in enumerate(sents):
        rel, en1, en2, text = sent.strip().split('\t', 3)
        examples.append(NREExample(str(i), text, en1, en2, rel))
        #idx += 1
    return examples

#num_labels = 9
num_labels = 9
subrels = [
        'ldcOnt:Evaluate.Deliberateness.Deliberate', 'ldcOnt:Evaluate.Legitimacy.Illegitimate', 'ldcOnt:Evaluate.Sentiment.Negative', 
        'ldcOnt:GeneralAffiliation.ArtifactPoliticalOrganizationReligiousAffiliation', 'ldcOnt:GeneralAffiliation.MemberOriginReligionEthnicity',
        'ldcOnt:GeneralAffiliation.OrganizationPoliticalReligiousAffiliation', 'ldcOnt:GeneralAffiliation.OrganizationWebsite.OrganizationWebsite', 'ldcOnt:GeneralAffiliation.Sponsorship',
        'ldcOnt:Information.Color.Color', 'ldcOnt:Information.Make.Make', 
        'ldcOnt:Measurement.Size',
        'ldcOnt:OrganizationAffiliation.EmploymentMembership', 'ldcOnt:OrganizationAffiliation.Founder.Founder', 'ldcOnt:OrganizationAffiliation.Leadership',
        'ldcOnt:PartWhole.Subsidiary',
        'ldcOnt:PersonalSocial.Role', 'ldcOnt:PersonalSocial.Unspecified',
        'ldcOnt:Physical.LocatedNear', 'ldcOnt:Physical.OrganizationHeadquarters.OrganizationHeadquarters', 'ldcOnt:Physical.Resident.Resident',
        'ldcOnt:ResponsibilityBlame.AssignBlame.AssignBlame', 'ldcOnt:ResponsibilityBlame.ClaimResponsibility.ClaimResponsibility', 'n/a'
    ]
num_labels = len(subrels)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
    do_lower_case=False)    
model_dir = '/home/xiangk/OPERA/NRE/type_models/NRE_BERT_ft_2e-5'
model_dir = '/home/xiangk/OPERA/NRE/subtype_models/NRE_BERT_ft_2e-5'
#sub_model_dir = ''
output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
output_config_file = os.path.join(model_dir, CONFIG_NAME)
config = BertConfig(output_config_file)
model = BERTPCNNNRE(config, num_labels=num_labels,
                            num_pos=62,
                            pos_dim=20,
                            rnn_size=128,
                            rnn_dropout=0.1,
                            filters=230)
model.load_state_dict(torch.load(output_model_file))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model.to(device)

def pred_nre(sents):
    
    eval_examples = read_sent(sents)
    label_list = get_labels()
    eval_features = convert_examples_to_features(
            eval_examples, label_list, 300, tokenizer, 30)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_left_pos = torch.tensor([f.lpos for f in eval_features], dtype=torch.long)
    all_right_pos = torch.tensor([f.rpos for f in eval_features], dtype=torch.long)
    all_lens = torch.tensor([f.len for f in eval_features], dtype=torch.long)
    all_pcnn_mask = torch.tensor([f.pcnn_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
        all_left_pos, all_right_pos, all_lens, all_pcnn_mask, all_label_ids)
    
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

    model.eval()
    pred_list = []
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, lpos_ids, rpos_ids, input_lens, pcnn_mask, label_ids = batch
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, lpos_ids, rpos_ids, input_lens,  pcnn_mask)
        logits = torch.nn.functional.softmax(logits, dim=1)
        probs = logits.detach().cpu().numpy()

        # x = logits
        # e_x = np.exp(x - np.max(x))
        # probs = e_x / e_x.sum(axis=0)
        # pred_label = np.argmax(logits, axis=1)
        pred_list.extend(probs)
    assert len(pred_list) == len(eval_dataloader)
    return pred_list

def read_nre_example(file):
    logger.info("LOOKING AT {}".format(file))
    examples = []
    with open(file, encoding='utf-8') as f:
        idx = 0
        for i, line in enumerate(f):
            rel, en1, en2, text = line.strip().split('\t', 3)
            examples.append(NREExample(str(idx), text, en1, en2, rel))
            #idx += 1
            #examples.append(NREExample(str(idx), text, en2, en1, rel))
            idx += 1
    return examples

def get_labels():
    return ['generalaffiliation', 'information', 'measurement', 'organizationaffiliation',
            'partwhole', 'personalsocial', 'physical', 'responsibilityblame', 'n/a']

def get_sub_lables():
    return [
        'Deliberateness', 'Legitimacy', 'Sentiment', 
        'ArtifactPoliticalOrganizationReligiousAffiliation', 'MemberOriginReligionEthnicity',
        'OrganizationPoliticalReligiousAffiliation', 'OrganizationWebsite', 'Sponsorship',
        'Color', 'Make', 
        'Size',
        'EmploymentMembership', 'Founder', 'Leadership',
        'Subsidiary',
        'Role', 'Unspecified',
        'LocatedNear', 'OrganizationHeadquarters', 'Resident',
        'AssignBlame', 'ClaimResponsibility', 'n/a'
    ]

def trun_pos(pos, left, right, limit):
    if pos >= left and pos <= right:
        pos = 0
    elif pos < left:
        pos = max(-limit, pos - left)
    else:
        pos = min(limit, pos - right)
    return pos + limit

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, pos_limit):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    idx = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        #print(example.text_a)
        label =  example.label
        label_id = label_map[label]
        tokens_a = tokenizer.tokenize(example.text_a)
        #labels_a = tokenize_label(example.text_a, tokens_a, example.label)
        # Account for [CLS] and [SEP] with "- 2"
        
            #labels_a = labels_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        
        #tokens_en1 = tokenizer.tokenize(example.en1) + ["[SEP]"]
        tokens_en1 = tokenizer.tokenize(example.en1)
        tokens_en2 = tokenizer.tokenize(example.en2)
        #tokens += tokens_en1 + ["[SEP]"]
        
        #tokens_en2 = tokenizer.tokenize(example.en2) + ["[SEP]"]
        #tokens += tokens_en2 + ["[SEP]"]
        if len(tokens) + len(tokens_en1) + len(tokens_en2) - 2 > max_seq_length:
            #print(tokens)
            tokens = tokens[:max_seq_length - (len(tokens_en1) + len(tokens_en2) - 2)]
            #print(tokens)
            if tokens.count("[MASK]") != 2:
                with open('too_long.txt', 'a') as f:
                    f.write(' '.join((str(ex_index), example.text_a, example.en1, example.en2)) + '\n')
                idx += 1
                continue
        #print(len(tokens), 'xk')
        pos_left = []
        tokens_len = len(tokens)
        pos_right = []
        #print(tokens)
        left_index, right_index = [i for i in range(len(tokens)) if tokens[i] == '[MASK]']
        left_len = len(tokens_en1)
        right_len = len(tokens_en2)
        pcnn_mask = [1] * (len(tokens_en1) + left_index) + [2] * (len(tokens_en2) + right_index - left_index - 1)
        pcnn_mask += [3] * (len(tokens) - right_index - 1)
        tokens = tokens[:left_index] + tokens_en1 + tokens[left_index+1:right_index] + tokens_en2 + tokens[right_index+1:]
        #print(len(tokens), len(pcnn_mask))
        assert len(tokens) == len(pcnn_mask)
        right_index += left_len - 1
        for i in range(len(tokens)):
            pos_left.append(trun_pos(i, left_index, left_index + left_len - 1, pos_limit))
            pos_right.append(trun_pos(i, right_index, right_index + right_len - 1, pos_limit))
        

        pos_padding = [pos_limit * 2 + 1] * (max_seq_length - len(tokens))
        #print(len(pos_padding), len(pos_left))
        pos_left += pos_padding
        pos_right += pos_padding
        segment_ids = [0] * len(tokens)
        #segment_ids += [1] * len(tokens_en1)
        #segment_ids += [1] * len(tokens_en2)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        #tokens += tokens_en1 + tokens_en2
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        pcnn_mask += padding
        #label_ids += padding
        #label_mask += padding
        #print(len(input_ids))
        if len(pcnn_mask) != max_seq_length:
            print(len(pcnn_mask), max_seq_length, tokens)
        assert len(pcnn_mask) == max_seq_length
        assert len(pos_left) == max_seq_length
        assert len(pos_right) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        #assert len(label_ids) == max_seq_length
        #assert len(label_mask) == max_seq_length

        #label_id = label_map[example.label]
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("en1 : %s" % " ".join(
                    [str(x) for x in tokens_en1]))
            logger.info("en2 : %s" % " ".join(
                    [str(x) for x in tokens_en2]))
            logger.info("pcnn mask: %s" % " ".join(
                    [str(x) for x in pcnn_mask]))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                    "left_pos: %s" % " ".join([str(x) for x in pos_left]))
            logger.info(
                    "right_pos: %s" % " ".join([str(x) for x in pos_right]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              lpos=pos_left,
                              rpos=pos_right,
                              label_id=label_id,
                              tokens_len=tokens_len,
                              pcnn_mask=pcnn_mask))
    #print(idx)
    return features

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file",
                        default='../../opera_rel_data/train_rel.txt',
                        type=str,
                        required=True,
                        help="train file path")
    parser.add_argument("--dev_file",
                        default='../../opera_rel_data/dev_rel.txt',
                        type=str,
                        required=True,
                        help="dev file path")
    
    parser.add_argument("--bert_model", default='bert-base-cased', type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--finetune_dir",
                        default='NRE_BERT',
                        type=str,
                        required=False,
                        help="The output")

    parser.add_argument("--output_dir",
                        default='NRE_BERT',
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
    parser.add_argument("--do_subtype",
                        action='store_true',
                        help="Whether to run for subtype.")
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
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--pos_dim",
                        default=20,
                        type=int,
                        help="position embedding dimension.")
    parser.add_argument("--filters",
                        default=230,
                        type=int,
                        help="pcnn output channel.")
    parser.add_argument("--rnn_size",
                        default=512,
                        type=int,
                        help="RNN hidden size.")
    parser.add_argument("--pos_limit",
                        default=30,
                        type=int,
                        help="limit of position.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-3,
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
                        default=2,
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
    #print(args)n
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
        do_lower_case=args.do_lower_case)
    if args.do_subtype:
        label_list = get_sub_lables()
    else:
        label_list = get_labels()
    num_labels = len(label_list)
    train_examples = read_nre_example(args.train_file)
    num_train_optimization_steps = None
    if args.do_train:
        #train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BERTPCNNNRE.from_pretrained(args.bert_model,
              cache_dir=cache_dir,
              num_labels=num_labels,
              num_pos=2 * args.pos_limit + 2,
              pos_dim=args.pos_dim,
              rnn_size=args.rnn_size,
              rnn_dropout=0.1,
              filters=args.filters)
    if not args.do_finetune:
        for param in model.bert.parameters():
            param.requires_grad = False
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
            train_examples, label_list, args.max_seq_length, tokenizer, args.pos_limit)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_left_pos = torch.tensor([f.lpos for f in train_features], dtype=torch.long)
        all_right_pos = torch.tensor([f.rpos for f in train_features], dtype=torch.long)
        all_lens = torch.tensor([f.len for f in train_features], dtype=torch.long)
        all_pcnn_mask = torch.tensor([f.pcnn_mask for f in train_features], dtype=torch.long)
        #all_label_masks = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_left_pos, all_right_pos, all_lens, all_pcnn_mask, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        model.train()
        for e in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # step, batch in enumerate(train_dataloader):

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lpos_ids, rpos_ids, input_lens, pcnn_mask, label_ids = batch
                # input_lens, sorted_idx = torch.sort(input_lens, descending=True)
                # input_ids = input_ids[sorted_idx]
                # input_mask = input_mask[sorted_idx]
                # segment_ids = segment_ids[sorted_idx]
                # lpos_ids = lpos_ids[sorted_idx]
                # rpos_ids = rpos_ids[sorted_idx]
                # label_ids = label_ids[sorted_idx]
                loss = model(input_ids, segment_ids, input_mask, lpos_ids,
                    rpos_ids, input_lens, pcnn_mask, label_ids)
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
                #if step % 500 == 0:
                    #logger.info('epoch %d step %d loss %.4f' % (e, step, tr_loss / nb_tr_steps))
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = BERTPCNNNRE(config, num_labels=num_labels,
                                    num_pos=2 * args.pos_limit + 2,
                                    pos_dim=args.pos_dim,
                                    rnn_size=args.rnn_size,
                                    rnn_dropout=0.1,
                                    filters=args.filters)
        model.load_state_dict(torch.load(output_model_file))
    else:
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        config = BertConfig(output_config_file)
        model = BERTPCNNNRE(config, num_labels=num_labels,
                                    num_pos=2 * args.pos_limit + 2,
                                    pos_dim=args.pos_dim,
                                    rnn_size=args.rnn_size,
                                    rnn_dropout=0.1,
                                    filters=args.filters)
        model.load_state_dict(torch.load(output_model_file))
        #model = BertForTokenClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_nre_example(args.dev_file)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, args.pos_limit)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_left_pos = torch.tensor([f.lpos for f in eval_features], dtype=torch.long)
        all_right_pos = torch.tensor([f.rpos for f in eval_features], dtype=torch.long)
        all_lens = torch.tensor([f.len for f in eval_features], dtype=torch.long)
        all_pcnn_mask = torch.tensor([f.pcnn_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
            all_left_pos, all_right_pos, all_lens, all_pcnn_mask, all_label_ids)
        
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        pred_list = []
        label_list = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lpos_ids, rpos_ids, input_lens, pcnn_mask, label_ids = batch
            # input_lens, sorted_idx = torch.sort(input_lens, descending=True)
            # input_ids = input_ids[sorted_idx]
            # input_mask = input_mask[sorted_idx]
            # segment_ids = segment_ids[sorted_idx]
            # lpos_ids = lpos_ids[sorted_idx]
            # rpos_ids = rpos_ids[sorted_idx]
            # label_ids = label_ids[sorted_idx]
            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, lpos_ids, rpos_ids, input_lens, pcnn_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask, lpos_ids, rpos_ids, input_lens,  pcnn_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            #print(logits, label_ids)
            tmp_eval_accuracy = accuracy(logits, label_ids)
            pred = np.argmax(logits, axis=1)
            pred_list.extend(pred)
            label_list.extend(label_ids)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        #print(label_list, pred_list)
        f1 = f1_score(label_list, pred_list, average='macro')  
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss/nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': loss,
                  'f1': f1}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))



if __name__ == "__main__":
    main()






