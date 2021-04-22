from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers import evaluation
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import argparse


def load_eval_sbert(path_eval_data, num_samples):
    df = pd.read_csv(path_eval_data)
    if num_samples>0:
        df = df.head(num_samples).copy()
    sentences1 = list(df.sentence1.values)
    sentences2 = list(df.sentence2.values)
    scores = [int(l) for l in list(df.label.values)]
    return sentences1, sentences2, scores

def load_train_sbert(path_train_data, num_samples):
    df = pd.read_csv(path_train_data)
    if num_samples>0:
        df = df.head(num_samples).copy()
    train_examples = [InputExample(texts=[s1, s2], label=int(l)) \
                for s1,s2,l in zip(list(df.sentence1.values), list(df.sentence2.values), list(df.label.values))]
    return train_examples

def main():
    parser = argparse.ArgumentParser(description='Start training with SBERT')
    parser.add_argument('--model_path',
                    type=str,
                    help='Path to trained model folder ./models/[MODEL_NAME]')
    parser.add_argument('--dataset',
                    type=str,
                    default='few_rel',
                    help='Name dataset')  
    parser.add_argument('--mask_method',
                    type=str,
                    default='bracket',
                    help='Type of masking')    
    parser.add_argument('--num_epochs',
                    type=int,
                    default=15,
                    help='Number epochs')                                
    parser.add_argument('--num_samples',
                    type=int,
                    default=-1,
                    help='Number of samples for test run, default -1 means all data')
    parser.add_argument('--max_seq_length',
                    type=int,
                    default=256,
                    help='Max token length for BERT')
    args = parser.parse_args()

    model_path = args.model_path
    dataset = args.dataset
    mask_method = args.mask_method
    num_samples = args.num_samples
    max_seq_length=args.max_seq_length
    num_epochs = args.num_epochs
    evaluation_steps = 1000 # Frequency of evaluation results
    warmup_steps = 1000 # warm up steps
    sentence_out_embedding_dimension = 256

    if model_path.endswith('/'):
        model_path = model_path[:-1]
    model_name = model_path.split('/')[-1]

    path_train_data = f'./data/train_samples/{dataset}_train_{mask_method}_train.csv'
    path_eval_data = f'./data/train_samples/{dataset}_val_{mask_method}_test.csv'
    if num_samples>0:
        model_save_path = f'./trained_models/{model_name}_sbert_bi_{dataset}_test/'
    else:
        model_save_path = f'./trained_models/{model_name}_sbert_bi_{dataset}/'
    ### Define the model
    word_embedding_model = models.Transformer(model_path, max_seq_length=max_seq_length)

    ### Add special tokens - this helps us add tokens like Doc or query or Entity1 / Entity2 
    # but in our case we already added that to the model prior
    #tokens = ["[DOC]", "[QRY]"]
    #word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
    #word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                        out_features=sentence_out_embedding_dimension, activation_function=nn.Tanh())
    # Model pipeline
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

    # Prep DataLoader
    train_examples = load_train_sbert(path_train_data, num_samples)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Prep Evaluator
    sentences1, sentences2, scores = load_eval_sbert(path_eval_data, num_samples)
    #evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    evaluator = evaluation.BinaryClassificationEvaluator(sentences1, sentences2, scores)
    #train_loss = losses.CosineSimilarityLoss(model)
    train_loss = losses.SoftmaxLoss(model, sentence_embedding_dimension= sentence_out_embedding_dimension, num_labels = 2)

    #Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=model_save_path)

if __name__ == '__main__':
    main()