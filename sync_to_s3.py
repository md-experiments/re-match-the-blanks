
import boto3
import os
from source.utils import get_keys,ProgressPercentage
import argparse

def main():
    parser = argparse.ArgumentParser(description='Archive models and upload to s3')
    parser.add_argument('--model_path',
                    type=str,
                    help='Path to trained model folder ./trained_models/[MODEL_NAME]')
    args = parser.parse_args()
    
    bucket_path = 're-matching-the-blanks'
    models_path = args.model_path
    model_name = models_path.split('/')[-1]
    # Remove checkpoints
    os.system(f'rm -r {models_path}/checkpoint-*')
    # ZIP model
    os.system(f'zip -r {models_path}.zip {models_path}')
    ### Upload to s3
    file_key = './aws_key_list.txt'
    AWS_ACCESS_KEY, AWS_SECRET_KEY = get_keys(file_key)
    s3client = boto3.client('s3', 
                    aws_access_key_id=AWS_ACCESS_KEY,
                    aws_secret_access_key=AWS_SECRET_KEY)
    s3client.upload_file(f'{models_path}.zip', bucket_path, f'models/{model_name}.zip',
                        Callback=ProgressPercentage(f'{models_path}.zip'))

if __name__=='__main__':
    main()