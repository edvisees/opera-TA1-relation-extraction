import torch
import json
import pickle
import sys
sys.path.append("pytorch-pretrained-bert/examples")
sys.path.append("..")
import os
import numpy as np
import importlib
import traceback
mod = importlib.import_module("run_nre")

#import main_mil
#from itertools import tee, izip
import argparse
from multiprocessing.dummy import Pool
import time
import pickle as pkl
import math
from numpy import size

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def rel_postprocessing(rel_prob, en1_type, en2_type, rel_dict, rel2id, sen):

    en1_type = en1_type.split('.')[0].split(':')[1].lower()
    en2_type = en2_type.split('.')[0].split(':')[1].lower()
    en1_en2 = ','.join((en1_type, en2_type))
    #print(rel_prob.shape)
    # if size(rel_prob) < 9:
    #     return len(rel_prob) - 1, 0, 0
    #rel_prob = np.append(rel_prob, 1e-5)
    #rel_prob = np.append(rel_prob, 1e-5)
    #rel_prob[8] += 0.1
    #rel_prob[26] += 0.2
    #rel_prob[0] += 0.05
    #print en1_en2
    if en1_en2 not in rel_dict:
        return len(rel_prob) - 1, 0, 0

    rels_candidate = rel_dict[en1_en2]
    mask = np.zeros_like(rel_prob)
    mask[-1] = 1.
    type_dict = {}
    for rc in rels_candidate:
        #rc_prob = float(rels_candidate[rc].split(',')[-1])
        mask[int(rc)] = 1.
    mask[0:3] = 0.
    if mask[rel2id['ldcOnt:Measurement.Size']] == 1 and en1_type == 'val':
        return rel2id['ldcOnt:Measurement.Size'], rel_dict[en1_en2][rel2id['ldcOnt:Measurement.Size']], 0.8 + np.random.rand() * 0.1
    # if mask[28] == 1 and en1_type =='NumericalValue':
    #     #print en1_en2
    #     return 28, rel_dict[en1_en2][28], 0.8 + np.random.uniform(0.101,0.180)
    # if mask[21] == 1:
    #     return 21, rel_dict[en1_en2][21], 0.8 + np.random.uniform(0.101,0.180)
    #if 
    mask_prob = rel_prob * mask
    rel = np.argmax(mask_prob)
    highest_prob = mask_prob[rel]

    if mask[rel2id['ldcOnt:GeneralAffiliation.Sponsorship']] == 1 and rel == len(rel_prob) - 1:
        if 'back' in sen or 'behind' in sen:
            return rel2id['ldcOnt:GeneralAffiliation.Sponsorship'], rel_dict[en1_en2][rel2id['ldcOnt:GeneralAffiliation.Sponsorship']], 0.5 - np.random.rand() * 0.1
        elif 'sponsor' in sen:
            return rel2id['ldcOnt:GeneralAffiliation.Sponsorship'], rel_dict[en1_en2][rel2id['ldcOnt:GeneralAffiliation.Sponsorship']], 0.8 - np.random.rand() * 0.1

    if rel == len(rel_prob) - 1:
        #sprint('xiangk')
        return len(rel_prob) - 1, 0, 0
    #print rels_candidate, rel, en1_type, en2_type, rel_prob
    #print rel
    #print rels_candidate
    #arg_split = rels_candidate[rel].split(',')
    #arg = ','.join((arg_split[0], arg_split[1]))
    return rel, rels_candidate[rel], 0.7 + np.random.uniform(0.05,0.102)

def rels_extract(ltf_data, file, rel_dicts, rel_dict, new_con):
    id2rel, rel2id = rel_dicts
    entity_file = os.path.join(ltf_data, file)
    out_rel = []
    num_pairs = 0
    with open(entity_file) as f:
        entites_json = json.load(f)
        acc_chars = 0
        for item in entites_json:
            cur_time = time.time()
            rels = []
            sen = item["inputSentence"]
            entities_info = [
                {'mention' : i['mention'],
                '@id' : i['@id'],
                'mention' : i['mention'],
                'char_begin' : i['char_begin'],
                'char_end' : i['char_end'],
                'type' : normalize_type(i['type']),
                "head_span_start": int(i["head_span"][0])
                }
                for i in item["namedMentions"]
            ]
            for i in item["nominalMentions"]:
                entities_info.append(
                    {'mention' : i['mention'],
                    '@id' : i['@id'],
                    'mention' : i['mention'],
                    'char_begin' : i['char_begin'],
                    'char_end' : i['char_end'],
                    'type' : normalize_type(i['type']),
                    "head_span_start": int(i["head_span"][0])
                    }
                )
            for i in item["fillerMentions"]:
                entities_info.append(
                    {'mention' : i['mention'],
                    '@id' : i['@id'],
                    'mention' : i['mention'],
                    'char_begin' : i['char_begin'],
                    'char_end' : i['char_end'],
                    'type' : normalize_type(i['type']),
                    "head_span_start": int(i["head_span"][0])
                    }
                )

            #entities_info = [(i['mention'], i["@id"], normalize_type(i['type']), i['head_span']) for i in item["namedMentions"]]
            #filler_info = [(i['mention'], i["@id"], normalize_type(i['type']), i['head_span']) for i in item["fillerMentions"]]
            #entities_info.extend(filler_info)
            entities = sorted(entities_info, key=lambda tupe: int(tupe['head_span_start']))
            #print(entities)
            num_pairs += len(entities)
            for i in range(len(entities) - 1):
                # en1 = entities[i][0]
                # en2 = entities[i+1][0]
                # en1_id = entities[i][1]
                # en2_id = entities[i+1][1]
                # en1_type = entities[i][2]
                # en2_type = entities[i+1][2]
                # span1 = entities[i][3][0]
                # span2 = entities[i+1][3][1]
                # real_span  = [span1, span2]
                en1 = entities[i]['mention']
                en2 = entities[i + 1]['mention']
                en1_id = entities[i]['@id']
                en2_id = entities[i + 1]['@id']
                en1_type =  entities[i]['type']
                en2_type =  entities[i + 1]['type']
                if en1 == en2:
                    continue

                #print(en1_type, en2_type)
                real_span = [entities[i]['char_begin'], entities[i + 1]['char_end']]
                # en1_begin, en1_end = entities[i]['char_begin'], entities[i]['char_end']
                # en2_begin, en2_end = entities[i + 1]['char_begin'], entities[i + 1]['char_end']
                # en1_begin -= acc_chars
                # en2_begin -= acc_chars
                # en1_end -= acc_chars
                # en2_end -= acc_chars
                en1_begin = sen.find(en1)
                en2_begin = sen.find(en2)
                en1_end = en1_begin + len(en1)
                en2_end = en2_begin + len(en2)
                # if sen[en1_begin : en1_end] != en1:
                #     print(acc_chars)
                #     print(en1_begin, en1_end)
                #     print(sen, sen[en1_begin : en1_end], en1, 'xk')
                # if sen[en2_begin : en2_end] != en2:
                #     print(sen, sen[en2_begin : en2_end], en2, 'xk')
                mask_sents = sen[:en1_begin] + ' [MASK] ' + sen[en1_end:en2_begin] + ' [MASK] '  + sen[en2_end:]
                
                #rel = 14
                #rel = main_mil.predict_no_label(model, sen, en1, en2)
                inp_sent = '\t'.join(('n/a', en1, en2, mask_sents))
                #print(inp_sent)
                if 'VAL' in en1_type:
                    en1_type = en1_type.split('.')[0].split(':')[1].lower()

                    for j in range(i+1, min(i+5, len(entities))):
                        en2 = entities[j]['mention']
                        en2_begin = sen.find(en2)
                        en2_end = en2_begin + len(en2)
                        en2_id = entities[j]['@id']
                        en2_type =  entities[j]['type']
                        en2_type = en2_type.split('.')[0].split(':')[1].lower()
                        en1_en2 = ','.join((en1_type, en2_type))
                        if ('%' not in sen[en1_begin:en2_end]) and ('percent' not in sen[en1_begin:en2_end]) \
                            and en1_en2 in new_con['ldcOnt:Measurement.Size.Count']:
                            rel = 'ldcOnt:Measurement.Size.Count'
                            real_span = [entities[i]['char_begin'], entities[j]['char_end']]
                            args_type = new_con[rel][en1_en2].split(',')
                            args_type = ['_'.join((rel, args_type[0])), '_'.join((rel, args_type[1]))]
                            arg_types_list = args_type
                            rel_json = { arg_types_list[0] : en1_id, arg_types_list[1] : en2_id, 'rel' : rel, 'score':0.9, 
                            'span': real_span, 'en1' : en1, 'en2' : en2}
                            rels.append(rel_json) 
                            break
                        elif ('%' in sen[en1_begin:en2_end] or 'percent' in sen[en1_begin:en2_end]) \
                            and en1_en2 in new_con['ldcOnt:Measurement.Size.Percentage']:
                            rel = 'ldcOnt:Measurement.Size.Percentage'
                            real_span = [entities[i]['char_begin'], entities[j]['char_end']]
                            args_type = new_con[rel][en1_en2].split(',')
                            args_type = ['_'.join((rel, args_type[0])), '_'.join((rel, args_type[1]))]
                            arg_types_list = args_type
                            rel_json = { arg_types_list[0] : en1_id, arg_types_list[1] : en2_id, 'rel' : rel, 'score':0.9, 
                            'span': real_span, 'en1' : en1, 'en2' : en2}
                            rels.append(rel_json) 
                            break
                    continue

                if 'GPE' in en1_type and 'PER' in en2_type  and entities[i+1]['char_begin'] - entities[i]['char_end'] < 5 :
                    rel = [0.] * len(id2rel)
                    rel[4] = 1.
                elif 'GPE' in en1_type and 'TTL' in en2_type and i + 2 < len(entities) \
                 and 'PER' in entities[i + 2]['type']:
                    #print('xiangk')
                    en2 = entities[i + 2]['mention']
                    en2_id = entities[i + 2]['@id']
                    en2_type =  entities[i + 2]['type']
                    rel = [0.] * len(id2rel)
                    rel[4] = 1.
                    rel, arg_types, prob = rel_postprocessing(rel, en1_type, en2_type, rel_dict, rel2id, sen[en1_end:en2_begin])
                    arg_types_list = arg_types.split(',')
                    rel_json = { arg_types_list[0] : en1_id, arg_types_list[1] : en2_id, 'rel' : id2rel[rel], 'score':prob, 
                            'span': real_span, 'en1' : en1, 'en2' : en2}
                    rels.append(rel_json)
                    continue
                else:
                    try:
                        rel = mod.pred_nre([inp_sent])
                    except:
                        continue
                    if len(rel) == 0:
                        continue
                    rel = rel[0]
                
                
                prob = 0.8
                rel, arg_types, prob = rel_postprocessing(rel, en1_type, en2_type, rel_dict, rel2id, sen[en1_end:en2_begin])
                #arg_types = 'dummy,dummy'
                #print(entities)
                # if id2rel[rel] == 'ldcOnt:Measurement.Size' and (entities[i+1]['char_begin'] - entities[i]['char_end'] > len(en1)):
                #     continue
                if id2rel[rel] == 'n/a':
                    continue
                rel = id2rel[rel]
                if rel == 'ldcOnt:Measurement.Size':
                    if '%' not in en1:
                        rel = 'ldcOnt:Measurement.Size.Count'
                    else:
                        rel =  'ldcOnt:Measurement.Size.Percentage'
                en1_type = en1_type.split('.')[0].split(':')[1].lower()
                en2_type = en2_type.split('.')[0].split(':')[1].lower()
                en1_en2 = ','.join((en1_type, en2_type))
                if en1_en2 not in new_con[rel]:
                    continue
                args_type = new_con[rel][en1_en2].split(',')
                args_type = ['_'.join((rel, args_type[0])), '_'.join((rel, args_type[1]))]
                #print(arg_types, 0.8)
                
                arg_types_list = args_type
                rel_json = { arg_types_list[0] : en1_id, arg_types_list[1] : en2_id, 'rel' : rel, 'score':prob, 
                            'span': real_span, 'en1' : en1, 'en2' : en2}
                rels.append(rel_json)
            #print(sen)
            acc_chars += len(sen)
            out_rel.append({'docID' : file, 'inputSentence' : sen, 'rels' : rels })
    return out_rel, num_pairs

def normalize_type(t):
    if t == 'GeopoliticalEntity':
        return 'gpe'
    elif t == 'Organization':
        return 'org'
    elif t == 'Location':
        return 'loc'
    elif t == 'Person':
        return 'per'
    elif t == 'Facility':
        return 'fac'
    elif t == 'Weapon':
        return 'wea'
    elif t == 'Vehicle':
        return 'veh'
    elif t == 'URL':
        return 'val'
    elif t == 'Title':
        return 'ttl'
    elif t == 'Time':
        return 'val'
    elif t == 'NumericalValue':
        return 'val'
    return t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ltf', type=str, help='ltf data folder')
    parser.add_argument('--out_folder', type=str, help='output folder')
    #parser.add_argument('file_name', type=str, help='file')
    args = parser.parse_args()
    ltf_data = args.ltf
    out_folder = args.out_folder
    

    #file_name = args.file_name
    #model_path = '/home/xiangk/LSTM-ER/pytorch_relation_extraction/checkpoints/model.pth'
    #model = torch.load(model_path)
    # id2rel = {}
    # with open('/home/xiangk/LSTM-ER/data/id2relation.txt') as f:
    #     for line in f:
    #         line_split = line.strip().split()
    #         id2rel[int(line_split[0])] = line_split[1]
    rels = ['generalaffiliation', 'information', 'measurement', 'organizationaffiliation',
            'partwhole', 'personalsocial', 'physical', 'responsibilityblame', 'n/a']
    subrels = [
        'ldcOnt:Evaluate.Deliberateness.Deliberate', 'ldcOnt:Evaluate.Legitimacy.Illegitimate', 'ldcOnt:Evaluate.Sentiment.Negative', 
        'ldcOnt:GeneralAffiliation.ArtifactPoliticalOrganizationReligiousAffiliation', 'ldcOnt:GeneralAffiliation.MemberOriginReligionEthnicity',
        'ldcOnt:GeneralAffiliation.OrganizationPoliticalReligiousAffiliation', 'ldcOnt:GeneralAffiliation.OrganizationWebsite.OrganizationWebsite', 'ldcOnt:GeneralAffiliation.Sponsorship',
        'ldcOnt:Information.Color.Color', 'ldcOnt:Information.Make.Make', 
        'ldcOnt:Measurement.Size',
        'ldcOnt:OrganizationAffiliation.EmploymentMembership', 'ldcOnt:OrganizationAffiliation.Founder.Founder', 'ldcOnt:OrganizationAffiliation.Leadership',
        'ldcOnt:PartWhole.Subsidiary',
        'ldcOnt:PersonalSocial.Role', 'ldcOnt:PersonalSocial.Role',
        'ldcOnt:Physical.LocatedNear', 'ldcOnt:Physical.OrganizationHeadquarters.OrganizationHeadquarters', 'ldcOnt:Physical.Resident.Resident',
        'ldcOnt:ResponsibilityBlame.AssignBlame.AssignBlame', 'ldcOnt:ResponsibilityBlame.ClaimResponsibility.ClaimResponsibility', 'n/a'
    ]
    new_con = {}
    with open('relations.tsv') as f:
        next(f)
        for line in f:
            line = line.split('\t')
            t, st, sst = line[1], line[3], line[5]
            if sst == 'n/a':
                ldc_type = 'ldcOnt:' + '.'.join((t, st))
            else:
                ldc_type = 'ldcOnt:' + '.'.join((t, st, sst))
            # if ldc_type in subrels:
            arg1_label = line[9]
            arg2_label = line[12]
            arg1_type = line[11]
            arg2_type = line[14]
            new_con[ldc_type] = {}
            label12 = ','.join((arg1_label, arg2_label))
            label21 = ','.join((arg2_label, arg1_label))
            for t1 in arg1_type.strip().split(','):
                t1 = t1.strip()
                for t2 in arg2_type.strip().split(','):
                    t2 = t2.strip()
                    type12 = ','.join((t1, t2))
                    type21 = ','.join((t2, t1))
                    
                    new_con[ldc_type][type12] = label12
                    new_con[ldc_type][type21] = label21



    id2rel = {}
    rel2id = {}
    for i, r in enumerate(subrels):
        id2rel[i] = r
        rel2id[r] = i
    #rel_dict = {}
    # with open('/home/xiangk/LSTM-ER/data/rel_constrain.pkl') as f:
    #     rel_dict = pkl.load(f)
    # print(rel_dict)
    # exit()
    # with open('/home/xiangk/LSTM-ER/data/weighted_rel_dict.pkl') as f:
    #     weighted_rel_dict = pkl.load(f)
    '''
    ontology = OntologyType()
    decisions = ontology.load_decision_tree()
    with StanfordCoreNLP('/home/xiangk/LSTM-ER/xiangyang_code/stanford-corenlp-full-2017-06-09/') as nlp:
        for file in os.listdir(ltf_data):
            if file.endswith('.xml'):
                #if file + '.json' in os.listdir('test_data/'):
                    # continue
                    #pass
                run_document(os.path.join(ltf_data, file), nlp, ontology, decisions, out_folder)
    '''
    with open('./new_subrel_con.pkl', 'rb') as f:
        rel_dict = pkl.load(f)
        #print(rel_dict)
    
    entity_type_conversion = {}
    weighted_rel_dict = None
    rel_dicts = [id2rel, rel2id]
    start_time = time.time()
    for file in os.listdir(ltf_data):
        #file_name = os.path.split(file_name)[-1] 
        try:
            rels, num_pairs = rels_extract(ltf_data, file, rel_dicts, rel_dict, new_con)
            with open(os.path.join(out_folder, file), 'w') as f:
                json.dump(rels, f, indent=4, sort_keys=True)
                #print file_name
                #exit()
        except Exception as err:
            sys.stderr.write("ERROR: Exception occured while processing " + file + " \n")
            traceback.print_exc()
    end_time = time.time()
    print(num_pairs/(end_time - start_time))
    print('total elapsed time: {}'.format(end_time - start_time))




