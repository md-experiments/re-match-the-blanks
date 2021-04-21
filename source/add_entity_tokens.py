import os
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import argparse
import torch

"""
python source/add_entity_tokens.py --model_name distilbert-base-uncased
"""

def main():
    parser = argparse.ArgumentParser(description='Add entity tokens to model definition')
    parser.add_argument('--model_name',
                        default="bert-base-uncased",
                        type=str,
                        help='Name huggingface model')
    parser.add_argument("--new_tokens_to_zero",
                        action="store_true", 
                        help="Store new token embeddings as zero"
    )
    args = parser.parse_args()
    new_tokens_to_zero = args.new_tokens_to_zero
    print(new_tokens_to_zero)
    print('Is false',new_tokens_to_zero==False)
    model_name = args.model_name

    if not os.path.exists('models'):
        os.mkdir('models')
    else:
        print('./models exists already')
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    new_tokens = ['<e1>','</e1>','<e2>','</e2>']
    if any(100==tokenizer.convert_tokens_to_ids(t) for t in new_tokens):
        print(len(tokenizer))  # 28996
        tokenizer.add_tokens(new_tokens)
        print(len(tokenizer))  # 28997
        model.resize_token_embeddings(len(tokenizer)) 
        
        # Set embeddings weigth to zero 
        if new_tokens_to_zero:
            for tok in new_tokens:
                tok_id = tokenizer.convert_tokens_to_ids(tok)
                model.embeddings.word_embeddings.weight[tok_id, :] = torch.zeros([model.config.hidden_size])
                
        if new_tokens_to_zero:
            method_name = 'mtb-zro'
        else:
            method_name = 'mtb-rnd'
        save_to_path = f'./models/{model_name}-{method_name}'

        model.save_pretrained(save_to_path)
        tokenizer.save_pretrained(save_to_path)
    else:
        print('Tokens already exist')
    
if __name__=='__main__':
    main()