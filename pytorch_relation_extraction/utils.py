# -*- coding: utf-8 -*-

import numpy as np
import time


class Sen2Train(object):
    '''
    load and preprocess data
    '''
    def __init__(self, max_len=80, limit=50, pos_dim=5, pad=1):

        self.max_len = max_len
        self.limit = limit
        #self.root_path = root_path
        self.pos_dim = pos_dim
        self.pad = pad
        wordlist = []
        wordlist.extend([word.strip('\n') for word in file('/home/xiangk/LSTM-ER/data/dict_comb.txt')])
        wordlist.append('BLANK')
        self.word2id = {j: i for i, j in enumerate(wordlist)}

    '''
    def parse_sen(self, sents):
        
        
        select_ent = []
        select_num = []
        select_sen = []
        select_pf = []
        select_pool = []
        for sent in sents:
            #print sen
            en1 = int(sent[sent[0]])
            en2 = int(sent[sent[1]])
            entities = [en1, en2]
            positions = []
            sentences = []
            entitiesPos = []
            positions.append(map(int, sent[0:2]))
            epos = map(lambda x: int(x) + 1, sent[0:2])
            epos.sort()
            entitiesPos.append(epos)
            sentences.append(map(int, sent[2:-1]))
            bag = [entities, 1, sentences, positions, entitiesPos]
            
            bag = self.get_sentence_feature([bag])
            select_ent.append(bag[0][0])
            select_num.append(bag[0][1])
            select_sen.append(bag[0][2][0])
            select_pf.append(bag[0][3][0])
            select_pool.append(bag[0][4][0])
        return [select_ent, select_num, select_sen, select_pf, select_pool]
    '''
    def parse_sen(self, sent):
        '''
        parse the records in data
        '''
        all_sens =[]
        all_labels =[]
        en1 = int(sent[sent[0]])
        en2 = int(sent[sent[1]])
        entities = [en1, en2]
        positions = []
        sentences = []
        entitiesPos = []
        positions.append(map(int, sent[0:2]))
        epos = map(lambda x: int(x) + 1, sent[0:2])
        epos.sort()
        entitiesPos.append(epos)
        sentences.append(map(int, sent[2:-1]))
        bag = [entities, 1, sentences, positions, entitiesPos]
        all_sens += [bag]
        bags_feature = self.get_sentence_feature(all_sens)
        return np.asarray(bags_feature)
        '''
        all_sens =[]
        all_labels =[]
        f = file(path)
        while 1:
            line = f.readline()
            if not line:
                break
            entities = map(int, line.split(' '))
            line = f.readline()
            bagLabel = line.split(' ')

            rel = map(int, bagLabel[0:-1])
            num = int(bagLabel[-1])
            positions = []
            sentences = []
            entitiesPos = []
            for i in range(0, num):
                sent = f.readline().split(' ')
                positions.append(map(int, sent[0:2]))
                epos = map(lambda x: int(x) + 1, sent[0:2])
                epos.sort()
                entitiesPos.append(epos)
                sentences.append(map(int, sent[2:-1]))
            bag = [entities, num, sentences, positions, entitiesPos]
            all_labels.append(rel)
            all_sens += [bag]

        f.close()
        bags_feature = self.get_sentence_feature(all_sens)

        return bags_feature, all_labels
        '''

    def get_sentence_feature(self, bags):
        '''
        : word embedding
        : postion embedding
        return:
        sen list
        pos_left
        pos_right
        '''
        update_bags = []

        for bag in bags:
            es, num, sens, pos, enPos = bag
            new_sen = []
            new_pos = []
            for idx, sen in enumerate(sens):
                new_pos.append(self.get_pos_feature(len(sen), pos[idx]))
                new_sen.append(self.get_pad_sen(sen))
            update_bags.append([es, num, new_sen, new_pos, enPos])

        return update_bags

    def get_pad_sen(self, sen):
        '''
        padding the sentences
        '''
        sen.insert(0, self.word2id['BLANK'])
        if len(sen) < self.max_len + 2 * self.pad:
            sen += [self.word2id['BLANK']] * (self.max_len +2 * self.pad - len(sen))
        else:
            sen = sen[: self.max_len + 2 * self.pad]

        return sen

    def get_pos_feature(self, sen_len, ent_pos):
        '''
        clip the postion range:
        : -limit ~ limit => 0 ~ limit * 2+2
        : -51 => 1
        : -50 => 1
        : 50 => 101
        : >50: 102
        '''

        def padding(x):
            if x < 1:
                return 1
            if x > self.limit * 2 + 1:
                return self.limit * 2 + 1
            return x

        if sen_len < self.max_len:
            index = np.arange(sen_len)
        else:
            index = np.arange(self.max_len)

        pf1 = [0]
        pf2 = [0]
        pf1 += map(padding, index - ent_pos[0] + 2 + self.limit)
        pf2 += map(padding, index - ent_pos[1] + 2 + self.limit)

        if len(pf1) < self.max_len + 2 * self.pad:
            pf1 += [0] * (self.max_len + 2 * self.pad - len(pf1))
            pf2 += [0] * (self.max_len + 2 * self.pad - len(pf2))
        return [pf1, pf2]





def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def save_pr(out_dir, name, epoch, pre, rec, fp_res=None, opt=None):
    if opt is None:
        out = file('{}/{}_{}_PR.txt'.format(out_dir, name, epoch + 1), 'w')
    else:
        out = file('{}/{}_{}_{}_PR.txt'.format(out_dir, name, opt, epoch + 1), 'w')

    if fp_res is not None:
        fp_out = file('{}/{}_{}_FP.txt'.format(out_dir, name, epoch + 1), 'w')
        for idx, r, p in fp_res:
            fp_out.write('{} {} {}\n'.format(idx, r, p))
        fp_out.close()

    for p, r in zip(pre, rec):
        out.write('{} {}\n'.format(p, r))

    out.close()


def eval_metric(true_y, pred_y, pred_p):
    '''
    calculate the precision and recall for p-r curve
    reglect the NA relation
    '''
    assert len(true_y) == len(pred_y)
    positive_num = len([i for i in true_y if i[0] > 0])
    index = np.argsort(pred_p)[::-1]

    tp = 0
    fp = 0
    fn = 0
    all_pre = [0]
    all_rec = [0]
    fp_res = []

    for idx in range(len(true_y)):
        i = true_y[index[idx]]
        j = pred_y[index[idx]]

        if i[0] == 0:  # NA relation
            if j > 0:
                fp_res.append((index[idx], j, pred_p[index[idx]]))
                fp += 1
        else:
            if j == 0:
                fn += 1
            else:
                for k in i:
                    if k == -1:
                        break
                    if k == j:
                        tp += 1
                        break

        if fp + tp == 0:
            precision = 1.0
        else:
            precision = tp * 1.0 / (tp + fp)
        recall = tp * 1.0 / positive_num
        if precision != all_pre[-1] or recall != all_rec[-1]:
            all_pre.append(precision)
            all_rec.append(recall)

    print("tp={}; fp={}; fn={}; positive_num={}".format(tp, fp, fn, positive_num))
    return all_pre[1:], all_rec[1:], fp_res
