import pandas as pd
import re
from gensim import corpora, models, similarities
from gensim.utils import tokenize

def save_tfidf_data(csv_filename):
    dataset = pd.read_csv('1639156572535.csv', sep=';')
    themes = dataset.groupby(['cat_id', 'cat_name', 'theme_name', 'theme_id',])['comment_text'].apply(' '.join).reset_index()
    
    tokens = list()
    corpuses = list()

    for theme_id, theme_text in themes[['theme_id', 'comment_text']].values:
        tokens.append({'theme_id': theme_id, 'tokens': list(tokenize(theme_text, lowercase=True,deacc=True))})

    dictionary = corpora.Dictionary([t['tokens'] for t in tokens])

    for t in tokens:
        corpuses.append(dictionary.doc2bow(t['tokens']))
    
    f_cnt = len(dictionary.token2id)
    tfidf = models.TfidfModel(corpuses)
    index = similarities.SparseMatrixSimilarity(tfidf[corpuses], num_features = f_cnt)

    dictionary.save('words.dictionary')
    tfidf.save('tfidf.model')
    index.save('tfidf.similarity')

def load_tfidf_data():
    result = dict()
    result['themes'] = pd.read_csv('1639156572535.csv', sep=';').groupby(['cat_id', 'cat_name', 'theme_name', 'theme_id',])['comment_text'].apply(' '.join).reset_index()
    result['dictionary'] = corpora.Dictionary.load('words.dictionary')
    result['f_cnt'] = len(result['dictionary'].token2id)
    result['tfidf'] = models.TfidfModel.load('tfidf.model')
    result['index'] = similarities.SparseMatrixSimilarity.load('tfidf.similarity')
    return result

def get_themes_indexes(tfidf_data, comment_text):
    vect = tfidf_data['dictionary'].doc2bow(list(tokenize(comment_text, lowercase=True,deacc=True)))
    tfidf_vect = list(tfidf_data['index'][tfidf_data['tfidf'][vect]])
    result = tfidf_data['themes'][['cat_id', 'cat_name', 'theme_id', 'theme_name']]
    result['tfidf_similarity'] = tfidf_vect
    return list(result.to_records(index=False))

if __name__ == '__main__':
    try:
        tfidf_data = load_tfidf_data()
    except FileNotFoundError as e:
        save_tfidf_data('1639156572535.csv')
        tfidf_data = load_tfidf_data()
    while True:
        print(get_themes_indexes(tfidf_data, input('>>>>')))