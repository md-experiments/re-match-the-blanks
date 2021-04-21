import requests
import json
from source.utils import create_path

def get_few_rel_data():
    file_names = ['val_wiki','train_wiki']
    create_path('./data/few_rel')
    for file_name in file_names:
        url = f'https://raw.githubusercontent.com/thunlp/FewRel/master/data/{file_name}.json'

        response = requests.get(url)
        data = response.json()
        with open(f'./data/few_rel/{file_name}.json','w') as d:
            json.dump(data,d)

if __name__ == '__main__':
    get_few_rel_data()