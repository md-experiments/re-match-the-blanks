from utils import create_path
import argparse
"""
python source/script_writers.py --masking_method mask_one
"""
class DictToClass:
    def __init__(self, d):
        for ky, vl in d.items():
            setattr(self, ky, vl)

def create_run_glue_script(p):
    p = DictToClass(p)
    c = f"python run_glue.py \
       --model_name_or_path {p.model_path} \
       --do_predict \
       --max_seq_length {p.max_len} \
       --num_train_epochs {p.tr_ep} \
       --train_file {p.file}_train.csv \
       --validation_file {p.file}_val.csv"
    
    if not p.predict_mode:
       c = c + f" --test_file {p.file_test}_test.csv --do_train --do_eval"
    else:
        c = c + f" --test_file {p.predict_file}"

    if p.test_mode:
        c = c + f" --output_dir {p.output_dir}_test \
        --max_train_samples {p.nr_test}\
        --max_val_samples {p.nr_test}\
        --max_test_samples {p.nr_test}"
    elif p.predict_mode:
        c = c + f" --output_dir {p.output_dir}_pred"
    else:
        c = c + f" --output_dir {p.output_dir}"
    print(c)

def create_sbert_script(p):
    p = DictToClass(p)
    c = f"python train_sbert.py \
            --model_path {p.model_path} \
            --max_seq_length {p.max_len} \
            --dataset {p.dataset} \
            --num_epochs {p.tr_ep}"

    if p.test_mode:
        c = c + f" --num_samples {p.nr_test}"
    print(c)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create data by masking method')
    parser.add_argument('--masking_method',
                        default="bracket",
                        type=str,
                        help='Default: bracket, Alternatives: mask_one, ignore')
    parser.add_argument('--predict_file',
                        default="./",
                        type=str,
                        help='Default: bracket, Alternatives: mask_one, ignore')
    args = parser.parse_args()
    method = args.masking_method
    predict_file = args.predict_file
    model_name = 'distilbert-base-uncased-mtb-rnd'
    exp = f'fel_rel_{method}'
    create_path('./trained_models/')
    
    print('='*60+' TEST '+'='*60)
    p = dict(
        model_path = f'./models/{model_name}/',
        max_len = 128*2,
        tr_ep = 1,
        output_dir = f'./trained_models/{model_name}_{exp}',
        file = f'./data/train_samples/few_rel_train_{method}',
        file_test = f'./data/train_samples/few_rel_val_{method}',
        test_mode = True,
        predict_mode = False,
        nr_test = 10
    )

    print()
    create_run_glue_script(p)
    print()

    print('='*60+' PREDICT '+'='*60)
    p = dict(
        model_path = f'./models/{model_name}/',
        max_len = 128*2,
        tr_ep = 1,
        output_dir = f'./trained_models/{model_name}_{exp}',
        file = f'./data/train_samples/few_rel_train_{method}',
        file_test = f'./data/train_samples/few_rel_val_{method}',
        test_mode = False,
        predict_mode = True,
        predict_file = predict_file,
        nr_test = 10
    )

    print()
    create_run_glue_script(p)
    print()


    print('='*60+' REAL '+'='*60)
    p = dict(
        model_path = f'./models/{model_name}/',
        max_len = 128*2,
        tr_ep = 15,
        output_dir = f'./trained_models/{model_name}_{exp}/',
        file = f'./data/train_samples/few_rel_train_{method}',
        file_test = f'./data/train_samples/few_rel_val_{method}',
        predict_mode = False,
        test_mode = False,
        nr_test = 10
    )
    print()
    create_run_glue_script(p)


    print()
    print('='*60+' SBERT TEST '+'='*60)
    p = dict(
        model_path = f'./models/{model_name}/',
        max_len = 128*2,
        dataset = 'few_rel',
        tr_ep = 1,
        test_mode = True,
        nr_test = 10
    )
    print()
    create_sbert_script(p)

    

    print()
    print('='*60+' SBERT REAL '+'='*60)
    p = dict(
        model_path = f'./models/{model_name}/',
        max_len = 128*2,
        dataset = 'few_rel',
        tr_ep = 15,
        test_mode = False,
    )

    print()
    create_sbert_script(p)

