import os,sys
sys.path.append(os.getcwd())
from IPython.core.interactiveshell import InteractiveShell
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
InteractiveShell.ast_node_interactivity = "all"

news_data_raw_v1 = pd.read_csv('news_data.csv')


def get_content_frame(news_data_raw_v1):
    news_data_raw_v1.drop(['source', 'theme', 'published_on'], axis=1, inplace=True)  #
    # preprocess content - description
    content_frame = news_data_raw_v1[['_id', 'title', 'description']]
    content_frame['content'] = content_frame['description'] + ' ' + content_frame['title']
    return content_frame


content_frame = get_content_frame(news_data_raw_v1)


def tokenize_frame(content_frame):
    content_frame.loc[:, 'sentences'] = content_frame['content'].apply(lambda x: sent_tokenize(x))
    content_frame.loc[:, 'words'] = content_frame['content'].apply(lambda x: word_tokenize(x))
    return content_frame


content_frame = tokenize_frame(content_frame)


# content_series
# & -in the company's name -> P&G


def remove_punctuations(x):
    punctuations_to_remove = '!"#$%\'()*+,/:;<=>?@[\\]^_`{|}~'  # needed puctuations
    table = str.maketrans('', '', punctuations_to_remove)
    stripped = [w.translate(table) for w in x]
    return stripped


def lower_list(x):
    return [word.lower() for word in x]


content_frame.loc[:, 'stripped_w'] = content_frame['words'].apply(lambda x: remove_punctuations(x))
content_frame.loc[:, 'stripped_w_lower'] = content_frame['stripped_w'].apply(lambda x: lower_list(x))
content_frame_vn = content_frame

# extract orgs names

import spacy
nlp = spacy.load('en_core_web_sm')

def get_orgs(x):
    doc = nlp(x)
    ents = []
    if doc.ents:
        for ent in doc.ents:
            if ent.label_ == 'ORG' or ent.label_ == 'PERSON':
                ents.append(ent.text)
    else:
        pass
    #         print('No named entities found.')
    return ents


def get_orgs_list(y):
    t = [get_orgs(x) for x in y]
    flat_list = [item for sublist in t for item in sublist]
    return flat_list

def process_content(content_frame_vn):
    org_names_raw_v1 = content_frame_vn['sentences'].apply(lambda x: get_orgs_list(x)).to_frame(name='companies_list')
    org_names_raw_v1['companies_list'].apply(len).value_counts(dropna=False)
    org_names_raw_v1['_id'] = content_frame_vn['_id']
    return org_names_raw_v1


org_names_raw_v1 = process_content(content_frame_vn)

# search for

from fuzzywuzzy import process
import pandas as pd

master_company_raw_v1 = pd.read_csv('master_company_sheet.csv')
choices = master_company_raw_v1['name'].values


def get_company_name(x):
    if str(x).isalnum():
        name, score = process.extract(x, choices, limit=1)[0]
        if score > 70:
            return name
        tokens = word_tokenize(x)
        for curr_token in tokens:
            if str(curr_token).isalnum():
                name, score = process.extract(curr_token, choices, limit=1)[0]
                if score > 70:
                    return name


def get_companies_list(y):
    L = [get_company_name(x) for x in y if x != None]
    t = list(set([x for x in L if x is not None]))
    return t


def get_company_name_high_score(x):
    name, score = process.extract(x, choices, limit=1)[0]
    if score > 80:
        return name
    tokens = word_tokenize(x)
    for curr_token in tokens:
        name, score = process.extract(curr_token, choices, limit=1)[0]
        if score > 80:
            return name


def get_companies_list_hs(y):
    L = [get_company_name_high_score(x) for x in y if x != None]
    t = list(set([x for x in L if x is not None]))
    return t


org_names_raw_v1.loc[:, 'identified_companies'] = org_names_raw_v1['companies_list'].apply(lambda x: get_companies_list(x))
org_names_raw_v1['len_comp'] = org_names_raw_v1['identified_companies'].apply(len)
org_names_raw_v1
org_names_raw_v1['len_comp'].value_counts()

# stopwords=['company','bank','investment','technologies']

# org_names_raw_v1[org_names_raw_v1['len_comp']==0]['pos'].apply(lambda x:get_companies_list_hs(x))
# bugfixes:

org_names_raw_v1[['title', 'description']] = news_data_raw_v1[['title', 'description']]
org_names_raw_v1[org_names_raw_v1['len_comp'] == 0].to_csv('test.csv')
org_names_raw_v1[org_names_raw_v1['len_comp'] == 0]
choices = master_company_raw_v1['name'].values
choices
