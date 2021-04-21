from source.utils import create_path

class DictToClass:
    def __init__(self, d):
        for ky, vl in d.items():
            setattr(self, ky, vl)

def create_run_glue_script(p):
    p = DictToClass(p)
    c = f"python run_glue.py \
       --model_name_or_path {p.model_path} \
       --do_train \
       --do_eval \
       --do_predict \
       --max_seq_length {p.max_len} \
       --num_train_samples {p.tr_ep} \
       --output_dir {p.output_dir} \
       --train_file {p.file}_train.csv \
       --validation_file {p.file}_val.csv \
       --test_file {p.file}_test.csv"
    
    if p.test_mode:
        c = c + f" --max_train_samples {p.nr_test}\
        --max_val_samples {p.nr_test}\
        --max_test_samples {p.nr_test}"
    print(c)

if __name__ == '__main__':
    model_name = 'distilbert-base-uncased-mtb-rnd'
    exp = 'fel_rel_bracket'
    create_path('./trained_models/')
    
    p = dict(
        model_path = f'./models/{model_name}/',
        max_len = 128*2,
        tr_ep = 15,
        output_dir = f'./trained_models/{model_name}_{exp}/',
        file = './data/train_samples/few_rel_train_bracket',
        test_mode = True,
        nr_test = 10
    )
    create_run_glue_script(p)
