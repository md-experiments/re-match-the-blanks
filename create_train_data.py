import random
import json
import os
import pandas as pd
from source.data_prep import sentence_masker
from source.utils import flatten_list, create_path
import argparse
import re
"""
python create_train_data.py --masking_method mask_one
python create_train_data.py --masking_method bracket --dataset sem_eval
"""

def read_sem_eval(path_file):
    with open(path_file, 'r') as d:
        ls_entries = d.read()

    split_lines = [[ln.split('\t')[0]] + ln.split('\t')[1].split('\n') for ln in ls_entries.split('\n\n')[:-1]]
    df = pd.DataFrame(split_lines, columns = ['i','sentence','label','comment']).set_index('i')
    df['sentence'] = df['sentence'].apply(lambda x:x[1:-1]).astype(str)
    return df

def create_sem_eval_dataset(df_semeval, file_name, save_path, method, ):
    assert('sem_eval' in file_name)
    random_state = 1701
    label_pos = 1
    label_neg = 0
    entity_tokens = {
        'h': ['<e1>','</e1>'],
        't': ['<e2>','</e2>']
    }
    col_nm = ['sentence1','sentence2','label','relation']
    list_rels = list(df_semeval.label.unique())
    def this_masker(sent_sample,entity_tokens = entity_tokens, method = method):
        # currently semeval comes directly with bracket method applied need to do extra work to apply to other methods
        assert(method in ['bracket','ignore'])
        if method == 'ignore':
            res = re.sub('<[/]?e[1,2]>','',sent_sample)
        elif method == 'bracket':
            res = sent_sample
        return res


    df_rel = pd.DataFrame([], columns = col_nm)

    for i, rel in enumerate(list_rels):
        # MATCHING RELATIONSHIPS
        # Split sentences half way and 
        all_sents_rel = list(df_semeval[df_semeval.label==rel].sentence.values)
        half_way = int(len(all_sents_rel) / 2)

        mask_set_match = [[this_masker(l),this_masker(r),label_pos, rel] \
                        for l,r in zip(all_sents_rel[:half_way], all_sents_rel[half_way:])]
        df_rel_pos_i = pd.DataFrame(mask_set_match, columns = col_nm)
        df_rel = df_rel.append(df_rel_pos_i, ignore_index=True)
        
        # MISMATCHING RELATIONSHIPS (NEGATIVE SAMPLES) 
        # Add the same number of negative examples from outside of the sentence pair
        all_sents_ex_current = list(df_semeval[df_semeval.label!=rel].sentence.values)
        
        random.seed(i)
        sample_inner = random.sample(all_sents_rel,k=half_way)
        sample_negative = random.sample(all_sents_ex_current,k=half_way)

        mask_set_mismatch = [[this_masker(l),this_masker(r),label_neg, rel] \
                            for l,r in zip(sample_inner, sample_negative)]
        df_rel_neg_i = pd.DataFrame(mask_set_mismatch, columns = col_nm)
        df_rel = df_rel.append(df_rel_neg_i, ignore_index=True)
    

    df_rel = df_rel.sample(frac=1,random_state=random_state).reset_index(drop=True)
    df_rel.to_csv(os.path.join(save_path,f'{file_name}_complete_{method}.csv'))
    
    extra_cols = ['relation']
    val_path = os.path.join(save_path,f'{file_name}.csv')
    df_rel.drop(extra_cols, axis=1).copy().to_csv(val_path, index=False)
    print('Saved file',val_path)

def create_few_rel_dataset(js_ls, file_name, save_path, split_data, method, ):
    random_state = 1701
    label_pos = 1
    label_neg = 0
    entity_tokens = {
        'h': ['<e1>','</e1>'],
        't': ['<e2>','</e2>']
    }
    col_nm = ['sentence1','sentence2','label','relation']
    list_rels = list(js_ls)
    def this_masker(sent_sample,entity_tokens = entity_tokens, method = method):
        return sentence_masker(sent_sample,entity_tokens, method = method)

    df_rel = pd.DataFrame([], columns = col_nm)

    for i, rel in enumerate(list_rels):
        # MATCHING RELATIONSHIPS
        # Split sentences half way and 
        all_sents_rel = js_ls[rel]
        half_way = int(len(all_sents_rel) / 2)

        mask_set_match = [[this_masker(l),this_masker(r),label_pos, rel] \
                        for l,r in zip(all_sents_rel[:half_way], all_sents_rel[half_way:])]
        df_rel_pos_i = pd.DataFrame(mask_set_match, columns = col_nm)
        df_rel = df_rel.append(df_rel_pos_i, ignore_index=True)
        
        # MISMATCHING RELATIONSHIPS (NEGATIVE SAMPLES) 
        # Add the same number of negative examples from outside of the sentence pair
        all_sents_ex_current = flatten_list([js_ls[r] for r in list_rels if r != rel])
        
        random.seed(i)
        sample_inner = random.sample(all_sents_rel,k=half_way)
        sample_negative = random.sample(all_sents_ex_current,k=half_way)

        mask_set_mismatch = [[this_masker(l),this_masker(r),label_neg, rel] \
                            for l,r in zip(sample_inner, sample_negative)]
        df_rel_neg_i = pd.DataFrame(mask_set_mismatch, columns = col_nm)
        df_rel = df_rel.append(df_rel_neg_i, ignore_index=True)
    
    df_rel = df_rel.sample(frac=1,random_state=random_state).reset_index(drop=True)
    df_rel.to_csv(os.path.join(save_path,f'{file_name}_complete_{method}.csv'))
    
    extra_cols = ['relation']
    if split_data:
        len_data = len(df_rel)
        train_len = int(0.8*len_data)
        #test_len = int(0.2*len_data)

        df_train = df_rel.iloc[:train_len].drop(extra_cols, axis=1).copy()
        df_val = df_rel.iloc[train_len:].drop(extra_cols, axis=1).copy()
        
        train_path = os.path.join(save_path,f'{file_name}_{method}_train.csv')
        df_train.to_csv(train_path, index=False)
        print('Saved file',train_path)
        
        val_path = os.path.join(save_path,f'{file_name}_{method}_val.csv')
        df_val.to_csv(val_path, index=False)
        print('Saved file',val_path)
    else:
        df_train = df_rel.drop(extra_cols, axis=1).copy()
        test_path = os.path.join(save_path,f'{file_name}_{method}_test.csv')
        df_train.to_csv(test_path, index=False)  
        print('Saved file',test_path)


def main():
    parser = argparse.ArgumentParser(
        description='Create data by masking method')
    parser.add_argument('--masking_method',
                        default="bracket",
                        type=str,
                        help='Default: bracket, Alternatives: mask_one, ignore')
    parser.add_argument('--dataset',
                        default="few_rel",
                        type=str,
                        help='Default: bracket, Alternatives: mask_one, ignore')
    args = parser.parse_args()
    method = args.masking_method
    dataset = args.dataset
    save_path = './data/train_samples/'
    create_path(save_path)
    if dataset == 'few_rel':
        with open('./data/few_rel/train_wiki.json','r') as data:
            js_ls = json.load(data)

        create_few_rel_dataset(js_ls, file_name = 'few_rel_train', 
                    save_path = save_path, 
                    split_data = True, method = method, )

        with open('./data/few_rel/val_wiki.json','r') as data:
            js_ls = json.load(data)

        create_few_rel_dataset(js_ls, file_name = 'few_rel_val', 
                    save_path = save_path, 
                    split_data = False, method = method, )

    elif dataset == 'sem_eval':
        file_name = f'sem_eval_train_{method}_train'
        path_file = './data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
        df_semeval = read_sem_eval(path_file)
        create_sem_eval_dataset(df_semeval, file_name, save_path, method, )

        file_name = f'sem_eval_val_{method}_test'
        path_file = './data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
        df_semeval = read_sem_eval(path_file)
        create_sem_eval_dataset(df_semeval, file_name, save_path, method, )


if __name__ == '__main__':
    main()