import torch
import json
import pickle
import sys
import os
import numpy as np
sys.path.insert(0, './pytorch_relation_extraction')
import main_mil
from itertools import tee, izip
import argparse
from nltk.tokenize import word_tokenize
import cPickle as pkl
import time
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def run_document(fname, nlp, ontology, decisions, out_folder):
    sents, doc = read_ltf_offset(fname, nlp=nlp)
    out_doc = []
    out_rel = []
    model_path = './resources/xiang/LSTM-ER/pytorch_relation_extraction/checkpoints/model.pth'
    model = torch.load(model_path)
    id2rel = {}
    with open('./resources/xiang/LSTM-ER/data/id2relation.txt') as f:
        for line in f:
            line_split = line.strip().split()
            id2rel[int(line_split[0])] = line_split[1]
    #print id2rel
    for sid, sent in enumerate(sents):
        # print(sent.get_text())
        rels = []
        named_ents, ners, feats = extract_ner(sent)
        #print(len(named_ents), named_ents, [word.word for word in sent.words])
        sentence = [word.word for word in sent.words]
        entities = [item['headword'] for item in named_ents]
        for en1, en2 in pairwise(entities):
            rel = main_mil.predict_no_label(model, sentence, en1, en2)

            rel_json = {'docID' : fname, 'en1' : en1, 'en2' : en2, 'rel' : id2rel[rel-1]}
            rels.append(rel_json)
        nominals = extract_nominals(sent, nlp, ners)
        # print(nominals)
        fillers = extract_filler(sent, nlp, ners)
        # print(fillers)

        for mention, feat in zip(named_ents, feats):
            mention['fineGrainedType'] = ontology.lookup(mention['headword'])
            if mention['fineGrainedType'] == 'NULL':
                mention['fineGrainedType'] = infer_type(feat, decisions, root=normalize_type(mention['type']))
        for mention in nominals:
            mention['fineGrainedType'] = ontology.lookup(mention['headword'])

        out_doc.append({'docID': fname, 'inputSentence': sent.get_original_string(), 'offset': sent.begin, 'namedMentions': named_ents, 'nominalMentions': nominals, 'fillerMentions': fillers})
        out_rel.append({'docID' : fname, 'inputSentence' : sent.get_original_string(), 'rels' : rels })
    with open(fname + '.json', 'w') as f:
        json.dump(out_doc, f, indent=4, sort_keys=True)
    outfile = os.path.join(out_folder, os.path.split(fname)[-1])
    with open(out_file+ '_rel.json', 'w') as f:
        json.dump(out_rel, f, indent=4, sort_keys=True)



def normalize_type(t):
    if t == 'ORG':
        return 'Org'
    if t == 'LOC':
        return 'Loc'
    if t == 'WEA':
        return 'Weapon'
    if t == 'VEH':
        return 'Vehicle'
    return t

def sen2id(sen):
    with open('word_dict.pkl') as f:
        word_dict = pkl.load(f)
    id_list = []
    for w in sen:
        if w.capitalize() in word_dict:
            w = w.capitalize()
        elif w not in word_dict:
            w = 'UNK'
        id_list.append(word_dict.get(w))
    
    return id_list

def en_sen_combine(sen, entity1, entity2):
    sen = word_tokenize(sen)
    entity1 = max(word_tokenize(entity1), key=len)
    entity2 = max(word_tokenize(entity2), key=len)
    #print sen, entity1, entity2
    if entity1 == entity2:
        indices = [i for i, x in enumerate(sen) if  entity2 in x]
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
            en2_index = [i for i, x in enumerate(sen) if  entity2 in x][0]
        en_indices = [en1_index, en2_index]
        #sen = ' '.join((str(en1_index), str(en2_index),))
    #en_indices = map(str, en_indices)
    if en_indices[0] == en_indices[1] or len(sen) > 80:
        return [0, 1, 1, 3, 2]
    sen = sen2id(sen)

    sen.insert(0, en_indices[1])
    sen.insert(0, en_indices[0])
    #print sen, 'xiang'
    return sen


def rels_extract(ltf_data, file, model, id2rel):
    entity_file = os.path.join(ltf_data, file)
    out_rel = []

    with open(entity_file) as f:
        entites_json = json.load(f)
        for item in entites_json:
            cur_time = time.time()
            rels_list = []
            sen = item["inputSentence"]
            entities = [i['headword'] for i in item["namedMentions"]]
            entity_ids = [i["@id"] for i in item["namedMentions"]]
            entity1_list = []
            entity2_list = []
            entities_ids_list = []
            sens = []
            print (time.time() - cur_time, 'part1')
            cur_time = time.time()
            for i in range(len(entities) - 1):
                en1 = entities[i]
                en2 = entities[i+1]
                en1_id = entity_ids[i]
                en2_id = entity_ids[i+1]
                entity2_list.append(en2)
                entity1_list.append(en1)
                entities_ids_list.append((en1_id, en2_id))
                sens.append(en_sen_combine(sen, en1, en2))
            print (time.time() - cur_time, 'part2')
            cur_time = time.time()
            if len(sens) > 0:
                #rels = main_mil.predict_no_label(model, sens)
            #rels = main_mil.predict_no_label(model, sen, entity1_list, entity2_list)
                rels = np.random.randint(22, size=len(sens))
            for i in range(len(entity1_list)):
                rel_json = {'en1' : entities_ids_list[i][0], 'en2' : entities_ids_list[i][1], 'rel' : id2rel[rels[i]-1]}
                rels_list.append(rel_json)
            out_rel.append({'docID' : file, 'inputSentence' : sen, 'rels' : rels_list })
    return out_rel

if __name__ == '__main__':
    cur_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('ltf', type=str, help='ltf data folder')
    parser.add_argument('out_folder', type=str, help='output folder')
    args = parser.parse_args()
    ltf_data = args.ltf
    out_folder = args.out_folder
    model_path = './resources/xiang/LSTM-ER/pytorch_relation_extraction/checkpoints/model.pth'
    model = torch.load(model_path)
    id2rel = {}
    with open('./resources/xiang/LSTM-ER/data/id2relation.txt') as f:
        for line in f:
            line_split = line.strip().split()
            id2rel[int(line_split[0])] = line_split[1]
    '''
    ontology = OntologyType()
    decisions = ontology.load_decision_tree()
    with StanfordCoreNLP('./resources/xiang/LSTM-ER/xiangyang_code/stanford-corenlp-full-2017-06-09/') as nlp:
        for file in os.listdir(ltf_data):
            if file.endswith('.xml'):
                #if file + '.json' in os.listdir('test_data/'):
                    # continue
                    #pass
                run_document(os.path.join(ltf_data, file), nlp, ontology, decisions, out_folder)
    '''
    print time.time() - cur_time
    for file in os.listdir(ltf_data):
        print file

        rels = rels_extract(ltf_data, file, model, id2rel)
        print time.time() - cur_time
        with open(os.path.join(out_folder, file), 'w') as f:
            json.dump(rels, f, indent=4, sort_keys=True)
            exit()



