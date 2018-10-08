import os
import subprocess

import gensim
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize
from keras.utils import get_file
from sklearn.manifold import TSNE

figsize(10, 10)

_MODEL = 'GoogleNews-vectors-negative300.bin'


def download_google_news_model(destination='generated'):
    unzipped_destination_file = os.path.join(destination, _MODEL)
    if not os.path.exists(unzipped_destination_file):
        # get_file won't download the file twice if it finds it in its cache
        path = get_file(f'{_MODEL}.gz', f'https://s3.amazonaws.com/dl4j-distribution/{_MODEL}.gz')

        if not os.path.isdir(destination):
            os.mkdir(destination)

        unzip_file(path, unzipped_destination_file)

    return unzipped_destination_file


def unzip_file(input_zipped_file, destination):

    if not os.path.isdir(destination):
        with open(destination, 'wb') as fout:
            zcat = subprocess.Popen(['zcat'], stdin=open(input_zipped_file), stdout=fout)
            zcat.wait()


def a_is_to_b_as_c_is_to(model, a, b, c, top_n=1):
    # Ensures a, b and c are lists.
    a, b, c = map(lambda x: x if isinstance(x, list) else [x], (a, b, c))

    positives = b + c
    negatives = a
    results = model.most_similar(positive=positives, negative=negatives, topn=top_n)

    if len(results) > 0:
        if top_n == 1:
            return results[0][0]  # Return single element
        else:
            return [result[0] for result in results]

    return None


def visualize_word_embeddings(model, items):
    # Remember that, basically, a model is just a big lookup table from words to vectors.
    item_vectors = np.asarray([model[item] for item in items if item in model])

    # Normalize vectors.
    lengths = np.linalg.norm(item_vectors, axis=1)
    norm_vectors = (item_vectors.T / lengths).T

    tsne = TSNE(n_components=2, perplexity=10, verbose=2).fit_transform(norm_vectors)

    xs = tsne[:, 0]
    ys = tsne[:, 1]

    _, ax = plt.subplots()
    ax.scatter(xs, ys)  # Place points on canvas

    for item, x, y in zip(items, xs, ys):
        ax.annotate(item, (x, y), size=14)  # Annotate each point with the word it represents

    plt.show()


def get_most_similar_terms(model, terms):
    assert isinstance(terms, str) or isinstance(terms, list)

    terms = [terms] if isinstance(terms, str) else terms

    return model.most_similar(positive=terms)


if __name__ == '__main__':
    # First, let's download the model.
    unzipped = download_google_news_model()

    # Then, let's create a Word2Vec model using the downloaded parameters.
    gnews_model = gensim.models.KeyedVectors.load_word2vec_format(unzipped, binary=True)

    # Let's look at some similar words
    print('Most similar words to "espresso":')
    print(get_most_similar_terms(gnews_model, 'espresso'))

    print('Man is to Woman as King is to:')
    print(a_is_to_b_as_c_is_to(gnews_model, 'man', 'woman', 'king'))

    for country in {'Italy', 'China', 'Venezuela', 'Colombia'}:
        print(f'{a_is_to_b_as_c_is_to(gnews_model, "Germany", "Berlin", country)} is the capital of {country}.')

    for company in {'Google', 'IBM', 'Boeing', 'Microsoft', 'Walmart', 'Tesla'}:
        products = a_is_to_b_as_c_is_to(gnews_model, ['Starbucks', 'Apple'], ['Starbucks_coffee', 'iPhone'], company,
                                        top_n=3)
        print(f'{company} -> {", ".join(products)}')

    beverages = ['espresso', 'beer', 'vodka', 'wine', 'cola', 'tea']
    countries = ['Italy', 'Spain', 'Canada', 'Russia', 'USA', 'Argentina', 'Venezuela', 'Colombia']
    sports = ['soccer', 'handball', 'basketball', 'hockey', 'cycling', 'cricket', 'baseball']
    animals = ['dog', 'cat', 'parrot', 'seahorse', 'zebra', 'rhino', 'elephant', 'cheetah']

    items = beverages + countries + sports + animals
    print(len(items))

    # Expectation: We should see four clear clusters, one for beverage, one for countries and one for sports.
    visualize_word_embeddings(gnews_model, items)
