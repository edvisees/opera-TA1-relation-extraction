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
from multiprocessing.dummy import Pool
import time
import cPickle as pkl
import math
from numpy import size
import traceback

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

def rel_postprocessing(rel_prob, en1_type, en2_type, rel_dict):
   
    en1_en2 = ','.join((en1_type, en2_type))
    #print rel_prob.shape
    if size(rel_prob) < 27:
        return 0, 0, 0
    rel_prob = np.append(rel_prob, 1e-5)
    rel_prob = np.append(rel_prob, 1e-5)
    rel_prob[0] += 0.1
    rel_prob[26] += 0.2
    #rel_prob[0] += 0.05
    #print en1_en2
    if en1_en2 not in rel_dict:
        return 0, 0, 0
    rels_candidate = rel_dict[en1_en2]
    mask = np.zeros_like(rel_prob)
    mask[0] = 1
    type_dict = {}
    for rc in rels_candidate:
        rc_prob = float(rels_candidate[rc].split(',')[-1])
        mask[int(rc)] = rc_prob
    if mask[28] == 1 and en1_type =='NumericalValue':
        #print en1_en2
        return 28, rel_dict[en1_en2][28], 0.8 + np.random.uniform(0.101,0.180)
    if mask[21] == 1:
        return 21, rel_dict[en1_en2][21], 0.8 + np.random.uniform(0.101,0.180)
    mask_prob = rel_prob * mask
    rel = np.argmax(mask_prob)
    highest_prob = mask_prob[rel]
    if rel == 0:
        return 0, 0, 0
    #print rels_candidate, rel, en1_type, en2_type, rel_prob
    #print rel
    #print rels_candidate
    arg_split = rels_candidate[rel].split(',')
    arg = ','.join((arg_split[0], arg_split[1]))
    return rel, arg, 0.5 + np.random.uniform(0.05,0.102)


def rels_extract(ltf_data, file, model, id2rel, rel_dict):
    
    entity_file = os.path.join(ltf_data, file)
    out_rel = []
    with open(entity_file) as f:
        entites_json = json.load(f)
        for item in entites_json:
            cur_time = time.time()
            rels = []
            sen = item["inputSentence"]
            entities_info = [(i['headword'], i["@id"], normalize_type(i['type']), i['head_span']) for i in item["namedMentions"]]
            filler_info = [(i['mention'], i["@id"], normalize_type(i['type']), i['head_span']) for i in item["fillerMentions"]]
            entities_info.extend(filler_info)
            entities = sorted(entities_info, key=lambda tupe: int(tupe[3][0]))
            #entities = [i['headword'] for i in item["namedMentions"]]
            #entity_ids = [i["@id"] for i in item["namedMentions"]]
            #types = [i['type'] for i in item["namedMentions"]]
            #spans = [i['head_span'] for i in item["namedMentions"]]
            for i in range(len(entities) - 1):
                en1 = entities[i][0]
                en2 = entities[i+1][0]
                en1_id = entities[i][1]
                en2_id = entities[i+1][1]
                en1_type = entities[i][2]
                en2_type = entities[i+1][2]
                span1 = entities[i][3][0]
                span2 = entities[i+1][3][1]
                real_span  = [span1, span2]
                #rel = 14
                rel = main_mil.predict_no_label(model, sen, en1, en2)
                rel, arg_types, prob = rel_postprocessing(rel, en1_type, en2_type, rel_dict)
                if rel == 28 and (entities[i+1][3][0] - entities[i][3][1] > 5):
                    continue
                if rel == 0:
                    continue
                arg_types_list = arg_types.split(',')
                rel_json = { arg_types_list[0] : en1_id, arg_types_list[1] : en2_id, 'rel' : id2rel[rel-1], 'score':prob, 
                            'span': real_span}
                rels.append(rel_json)
            out_rel.append({'docID' : file, 'inputSentence' : sen, 'rels' : rels })
    return out_rel

def normalize_type(t):
    if t == 'GeopoliticalEntity':
        return 'GPE'
    elif t == 'Organization':
        return 'ORG'
    elif t == 'Location':
        return 'LOC'
    elif t == 'Person':
        return 'PER'
    elif t == 'Facility':
        return 'FAC'
    elif t == 'Weapon':
        return 'WEA'
    elif t == 'Vehicle':
        return 'VEH'
    elif t == 'URL':
        return 'URL'
    elif t == 'Title':
        return 'TITLE'
    elif t == 'Time':
        return 'Time'
    elif t == 'NumericalValue':
        return 'NumericalValue'
    return t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ltf', type=str, help='ltf data folder')
    parser.add_argument('out_folder', type=str, help='output folder')
    #parser.add_argument('file_name', type=str, help='file')
    args = parser.parse_args()
    ltf_data = args.ltf
    out_folder = args.out_folder
    #file_name = args.file_name
    model_path = './resources/xiang/LSTM-ER/pytorch_relation_extraction/checkpoints/model.pth'
    model = torch.load(model_path)
    id2rel = {}
    with open('./resources/xiang/LSTM-ER/data/id2relation.txt') as f:
        for line in f:
            line_split = line.strip().split()
            id2rel[int(line_split[0])] = line_split[1]
    #rel_dict = {}
    with open('./resources/xiang/LSTM-ER/data/rel_constrain.pkl') as f:
        rel_dict = pkl.load(f)
    with open('./resources/xiang/LSTM-ER/data/weighted_rel_dict.pkl') as f:
        weighted_rel_dict = pkl.load(f)
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
    entity_type_conversion = {}

    for file in os.listdir(ltf_data):
    #file_name = os.path.split(file_name)[-1] 
        rels = rels_extract(ltf_data, file, model, id2rel, weighted_rel_dict)
        try:
            with open(os.path.join(out_folder, file), 'w') as f:
                json.dump(rels, f, indent=4, sort_keys=True)
                #print file_name
                #exit()
        except Exception as err:
            sys.stderr.write("ERROR: Exception occured while processing " + file + " \n")
            traceback.print_exc()




