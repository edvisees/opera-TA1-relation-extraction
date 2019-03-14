# -*- coding: utf-8 -*-
from __future__ import absolute_import
import time

from config import opt
import models
import dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from utils import save_pr, now, eval_metric
from sklearn.metrics import f1_score, precision_score, recall_score
import cPickle as pkl
from utils import Sen2Train
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")

with open('word_dict.pkl') as f:
    word_dict = pkl.load(f)

def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

def sen2id(sen, word_dict=word_dict):
    #cur_time = time.time()

    
    #print cur_time - time.time()
    id_list = []
    for w in sen:
        if w.capitalize() in word_dict:
            w = w.capitalize()
        elif w not in word_dict:
            w = 'UNK'
        id_list.append(word_dict.get(w))
    
    return id_list

def test(**kwargs):
    pass


def train(**kwargs):

    kwargs.update({'model': 'PCNN_ONE'})
    opt.parse(kwargs)

    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # torch.manual_seed(opt.seed)
    model = getattr(models, 'PCNN_ONE')(opt)
    if opt.use_gpu:
        # torch.cuda.manual_seed_all(opt.seed)
        model.cuda()
        #  model = nn.DataParallel(model)

    # loading data
    DataModel = getattr(dataset, opt.data + 'Data')
    train_data = DataModel(opt.data_root, train=True)
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)

    test_data = DataModel(opt.data_root, train=False)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    print('train data: {}; test data: {}'.format(len(train_data), len(test_data)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), rho=1.0, eps=1e-6, weight_decay=opt.weight_decay)

    # train
    print("start training...")
    max_f1_score = -1.
    for epoch in range(opt.num_epochs):
        #print epoch
        total_loss = 0
        for idx, (data, label_set) in enumerate(train_data_loader):
            #print idx
            #print data[0]
            label = [l[0] for l in label_set]

            if opt.use_gpu:
                label = torch.LongTensor(label).cuda()
            else:
                label = torch.LongTensor(label)

            data = select_instance(model, data, label)
            
            model.batch_size = opt.batch_size

            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, Variable(label))
            
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]

        if epoch < -3:
            continue
        true_y, pred_y, pred_p = predict(model, test_data_loader)
        single_true_y = []
        for t_y in true_y:
            single_true_y.append(t_y[0])
        #print single_true_y
        f1score = f1_score(single_true_y, pred_y, average='macro')  
        precision = precision_score(single_true_y, pred_y, average='macro')
        recall = recall_score(single_true_y, pred_y, average='macro')
        f1score_class = f1_score(single_true_y, pred_y, average=None)  
        precision_class = precision_score(single_true_y, pred_y, average=None)
        recall_class = recall_score(single_true_y, pred_y, average=None)
        #print pred_y
        if f1score > max_f1_score:
            max_f1_score = f1score
            print('save the model')
            torch.save(model, opt.load_model_path)
        #print true_y[:10], pred_y[:10]
        '''
        all_pre, all_rec, fp_res = eval_metric(true_y, pred_y, pred_p)

        last_pre, last_rec = all_pre[-1], all_rec[-1]
        if last_pre > 0.24 and last_rec > 0.24:
            save_pr(opt.result_dir, model.model_name, epoch, all_pre, all_rec, fp_res, opt=opt.print_opt)
            print('{} Epoch {} save pr'.format(now(), epoch + 1))
            if last_pre > max_pre and last_rec > max_rec:
                print("save model")
                max_pre = last_pre
                max_rec = last_rec
                model.save(opt.print_opt)
        '''
        print(precision_class, recall_class, f1score_class)
        print('{} Epoch {}/{}: train loss: {}; test precision: {}, test recall {}, f1_score {}'.format(now(), epoch + 1, opt.num_epochs, total_loss, precision, recall, f1score))


def select_instance(model, batch_data, labels):

    model.eval()
    select_ent = []
    select_num = []
    select_sen = []
    select_pf = []
    select_pool = []
    for idx, bag in enumerate(batch_data):
        insNum = bag[1]
        label = labels[idx]
        max_ins_id = 0
        if insNum > 1:
            model.batch_size = insNum
            if opt.use_gpu:
                data = map(lambda x: Variable(torch.LongTensor(x).cuda()), bag)
            else:
                data = map(lambda x: Variable(torch.LongTensor(x)), bag)

            out = model(data)

            #  max_ins_id = torch.max(torch.max(out, 1)[0], 0)[1]
            max_ins_id = torch.max(out[:, label], 0)[1]

            if opt.use_gpu:
                max_ins_id = max_ins_id.data.cpu().numpy()[0]
            else:
                max_ins_id = max_ins_id.data.numpy()[0]

        max_sen = bag[2][max_ins_id]
        max_pf = bag[3][max_ins_id]
        max_pool = bag[4][max_ins_id]

        select_ent.append(bag[0])
        select_num.append(bag[1])
        select_sen.append(max_sen)
        select_pf.append(max_pf)
        select_pool.append(max_pool)

    if opt.use_gpu:
        data = map(lambda x: Variable(torch.LongTensor(x).cuda()), [select_ent, select_num, select_sen, select_pf, select_pool])
    else:
        data = map(lambda x: Variable(torch.LongTensor(x)), [select_ent, select_num, select_sen, select_pf, select_pool])
    model.train()
    return data

def predict_no_label(model, sen, entity1, entity2):
    #print(sen)
    model.eval()

    sen = word_tokenize(sen)
    entity1 = max(word_tokenize(entity1), key=len)
    entity2 = max(word_tokenize(entity2), key=len)

    #print sen, entity1, entity2
    if entity1 == entity2:
        indices = [i for i, x in enumerate(sen) if  entity2 in x]
        if len(indices) < 2:
            return 0
        en_indices = [indices[0], indices[1]]
        #sen = ' '.join((str(indices[0]), str(indices[1]), sen))
    else:
        try:
            en1_index = sen.index(entity1) 
        except ValueError:
            en1_index = [i for i, x in enumerate(sen) if  entity1 in x][0]
        try:
            en2_index = sen.index(entity2) 
        except ValueError:
            try:
                en2_index = [i for i, x in enumerate(sen) if  entity2 in x][0]
            except:
                return 0
        en_indices = [en1_index, en2_index]
        #sen = ' '.join((str(en1_index), str(en2_index),))
    #en_indices = map(str, en_indices)
    if en_indices[0] == en_indices[1] or len(sen) > 80:
        return 0

    sen = sen2id(sen)
    cur_time = time.time()
    sen.insert(0, en_indices[1])
    sen.insert(0, en_indices[0])
    #print sen, 'xiangk'
    sen2train = Sen2Train()

    bag = sen2train.parse_sen(sen)
    
    #print bag[0]
    if opt.use_gpu:
        data = map(lambda x: Variable(torch.LongTensor(x).cuda()), bag[0])
    else:
        data = map(lambda x: Variable(torch.LongTensor(x)), bag)
    
    out = model(data)
    #print time.time() - cur_time

    out = F.softmax(out, 1)
    label = torch.max(out, 1)
    return out.data.cpu().numpy()[0]
'''



def predict_no_label(model, sens):
    #print(sen)
    model.eval()
    
    sen2train = Sen2Train()
    bag = sen2train.parse_sen(sens)
    #print bag[0]
    if opt.use_gpu:
        data = map(lambda x: Variable(torch.LongTensor(x).cuda()), bag)
    else:
        data = map(lambda x: Variable(torch.LongTensor(x)), bag)
    out = model(data)
    out = F.softmax(out, 1)
    label = torch.max(out, 1)
    return label[1].data.cpu()

'''
def predict(model, test_data_loader):

    model.eval()

    pred_y = []
    true_y = []
    pred_p = []
    for idx, (data, labels) in enumerate(test_data_loader):
        true_y.extend(labels)
        for bag in data:
            insNum = bag[1]
            model.batch_size = insNum
            if opt.use_gpu:
                data = map(lambda x: Variable(torch.LongTensor(x).cuda()), bag)
            else:
                data = map(lambda x: Variable(torch.LongTensor(x)), bag)

            out = model(data)
            out = F.softmax(out, 1)

            max_ins_prob, max_ins_label = map(lambda x: x.data.cpu().numpy(), torch.max(out, 1))
            tmp_prob = -1.0
            tmp_NA_prob = -1.0
            pred_label = 0
            pos_flag = False

            for i in range(insNum):
                if pos_flag and max_ins_label[i] < 1:
                    continue
                else:
                    if max_ins_label[i] > 0:
                        pos_flag = True
                        if max_ins_prob[i] > tmp_prob:
                            pred_label = max_ins_label[i]
                            tmp_prob = max_ins_prob[i]
                    else:
                        if max_ins_prob[i] > tmp_NA_prob:
                            tmp_NA_prob = max_ins_prob[i]

            if pos_flag:
                pred_p.append(tmp_prob)
            else:
                pred_p.append(tmp_NA_prob)

            pred_y.append(pred_label)

    size = len(test_data_loader.dataset)
    assert len(pred_y) == size and len(true_y) == size

    model.train()
    
    return true_y, pred_y, pred_p


if __name__ == "__main__":
    import fire
    fire.Fire()
    #sen = 'Tom sits near Xiang'
    #en1 = 'Tom'
    #en2 = 'Xiang'
    #model_path = 'checkpoints/model.pth'
    #predict_no_label(model_path, sen.split(), en1, en2)

