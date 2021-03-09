import os
import sys

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


def remove_lt_3(x):
    return [w for w in x if len(w) > 3]


content_frame.loc[:, 'stripped_w'] = content_frame['words'].apply(lambda x: remove_punctuations(x))
content_frame.loc[:, 'stripped_w_lower'] = content_frame['stripped_w'].apply(lambda x: lower_list(x))
content_frame.loc[:, 'stripped_w_lower'] = content_frame['stripped_w_lower'].apply(lambda x: remove_lt_3(x))

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
    # org_names_raw_v1['companies_list'].apply(len).value_counts(dropna=False)
    org_names_raw_v1['_id'] = content_frame_vn['_id']
    org_names_raw_v1['stripped_w_lower'] = content_frame_vn['stripped_w_lower']
    return org_names_raw_v1


org_names_raw_v1 = process_content(content_frame_vn)

# search for

from fuzzywuzzy import process
import pandas as pd

master_company_raw_v1 = pd.read_csv('master_company_sheet.csv')
choices = master_company_raw_v1['name'].values


def get_company_name(x):
    if str(x) not in string.punctuation:
        name, score = process.extract(x, choices, limit=1)[0]
        if score > 70:
            return name
        tokens = word_tokenize(x)
        for curr_token in tokens:
            if str(curr_token) not in string.punctuation:
                name, score = process.extract(curr_token, choices, limit=1)[0]
                if score > 70:
                    return name


def get_companies_list(y):
    L = [get_company_name(x) for x in y if x != None]
    t = list(set([x for x in L if x is not None]))
    return t


org_names_raw_v1.loc[:, 'identified_companies'] = org_names_raw_v1['companies_list'].apply(lambda x: get_companies_list(x))
org_names_raw_v1['len_comp'] = org_names_raw_v1['identified_companies'].apply(len)
org_names_raw_v1
org_names_raw_v1['len_comp'].value_counts()
import string

org_names_raw_v1.head(2)

master_company_raw_v1 = pd.read_csv('master_company_sheet.csv')
master_company_names = master_company_raw_v1['name'].values


def search_company_in_words(stripped_values):
    choices = stripped_values
    for curr_company in master_company_names:
        name, score = process.extract(curr_company, choices, limit=1)[0]
        if score > 70:
            print(name, score, curr_company)
            return curr_company


org_names_raw_v1[['title', 'description']] = news_data_raw_v1[['title', 'description']]
org_names_raw_v1[org_names_raw_v1['len_comp'] == 0].to_csv('test.csv')
choices = master_company_raw_v1['name'].values

run_op_v1 = org_names_raw_v1[['_id', 'identified_companies']]
run_op_v1.to_csv('run_op_v1.csv', index=False)

marked_data_vn = pd.read_csv('marked_data_vn.csv')
marked_data_vn.head(2)
score_raw_v1 = marked_data_vn.merge(run_op_v1, how='left')
from ast import literal_eval

try:
    score_raw_v1['identified_companies'] = score_raw_v1['identified_companies'].apply(literal_eval)
except ValueError:  # catch if already evaluated as string
    pass

try:
    score_raw_v1['companies'] = score_raw_v1['companies'].apply(literal_eval)
except ValueError:  # catch if already evaluated as string
    pass

score_raw_v1.loc[:, 'companies'] = score_raw_v1['companies'].apply(lambda x: sorted(x))
score_raw_v1.loc[:, 'identified_companies'] = score_raw_v1.apply(lambda x: sorted(x['identified_companies']), axis=1)
score_raw_v1[['_id', 'companies', 'identified_companies']].head(5)

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

mlb = MultiLabelBinarizer()
mlb.fit(master_company_raw_v1['name'].values.reshape(-1, 1))
mlb.classes_

master_company_raw_v1 = pd.read_csv('master_company_sheet.csv')


def get_score(score_raw_v1):
    score = f1_score(mlb.transform(score_raw_v1['companies'].values),
                     mlb.transform(score_raw_v1['identified_companies'].values),
                     average='macro', zero_division=0)
    return score


get_score(score_raw_v1)
