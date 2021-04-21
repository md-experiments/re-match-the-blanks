import random
import json
import os
import pandas as pd
from source.data_prep import sentence_masker
from source.utils import flatten_list, create_path




def create_few_rel_dataset(js_ls, file_name, save_path, split_data, method, ):
    random_state = 1701
    label_pos = 1
    label_neg = 0
    entity_tokens = {
        'h': ['<E1>','</E1>'],
        't': ['<E2>','</E2>']
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
        
        df_train.to_csv(os.path.join(save_path,f'{file_name}_{method}_train.csv'), index=False)
        df_val.to_csv(os.path.join(save_path,f'{file_name}_{method}_val.csv'), index=False)
    else:
        df_train = df_rel.drop(extra_cols, axis=1).copy()
        df_train.to_csv(os.path.join(save_path,f'{file_name}_{method}_test.csv'), index=False)  


def main():
    method = 'bracket'
    save_path = './data/train_samples/'
    create_path(save_path)
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

if __name__ == '__main__':
    main()