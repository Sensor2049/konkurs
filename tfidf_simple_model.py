import pandas as pd
import re
from gensim import corpora, models, similarities
from gensim.utils import tokenize

def save_tfidf_data(csv_filename):
    dataset = pd.read_csv('1639156572535.csv', sep=';')
    themes = dataset.groupby(['cat_id', 'cat_name', 'theme_name', 'theme_id',])['comment_text'].apply(' '.join).reset_index()
    tokens = dict()
    corpuses = dict()

    for theme_id, theme_text in themes[['theme_id', 'comment_text']].values:
        tokens[theme_id] = list(tokenize(theme_text, lowercase=True,deacc=True))

    dictionary = corpora.Dictionary(list(tokens.values()))

    for i in range(len(tokens)):
        corpuses[i] = dictionary.doc2bow(tokens[i] if i in tokens else [])

    tfidf = models.TfidfModel(list(corpuses.values()))
    index = similarities.SparseMatrixSimilarity(tfidf[list(corpuses.values())], num_features = f_cnt)

    dictionary.save('words.dictionary')
    tfidf.save('tfidf.model')
    index.save('tfidf.similarity')

def load_tfidf_data():
    result = dict()
    result['themes'] = pd.read_csv('1639156572535.csv', sep=';')[['cat_id', 'theme_id']]
    result['dictionary'] = corpora.Dictionary.load('words.dictionary')
    result['f_cnt'] = len(result['dictionary'].token2id)
    result['tfidf'] = models.TfidfModel.load('tfidf.model')
    result['index'] = similarities.SparseMatrixSimilarity.load('tfidf.similarity')
    return result

def get_category_and_theme_id(tfidf_data, comment_text):
    vect = tfidf_data['dictionary'].doc2bow(tokenize(comment_text))
    result = list(tfidf_data['index'][tfidf_data['tfidf'][vect]])
    pos = result.index(max(result))
    return list(tfidf_data['themes'][['cat_id', 'theme_id']][tfidf_data['themes'].theme_id == pos].values[0])

if __name__ == '__main__':
    tfidf_data = load_tfidf_data()
    while True:
        print(get_category_and_theme_id(tfidf_data, input('>>>>')))