### This code is for finding similar books to the one user requests
### Useful in case there are no existing ratings for the user

import pandas as pd
from scipy.linalg import svd
import scipy
from memoization import cached

def cosine_similarity(v, u):
    return (v @ u) / (scipy.linalg.norm(v) * scipy.linalg.norm(u))

def clean_duplicate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apparently there are cases of the same user giving different
    rating for the book. It can be due to different language,
    edition, etc.

    Later analysis of the data indicates the reason of such problem.
    Because the data was randomly-generated, and created gibberish cases
     (e.g. the same books switch authors and genre)
    Let's pretend doing something smart, and instead of dropping these
    reviews, in case of duplicated reviews by the same person, we
    keep only the one for the book of the latest publish_year
    """
    dupl_reviews = df.groupby(by=['book_id', 'reader_id']).count()
    dupl_reviews = dupl_reviews[dupl_reviews['book_rating'] > 1]
    df_dupl = df[df['book_id'].isin([el[0] for el in dupl_reviews.index.values])
                 & df['reader_id'].isin([el[1] for el in dupl_reviews.index.values])]

    df_dupl.sort_values(by=['publish_year'], ascending=False, inplace=True)
    to_keep = df_dupl.drop_duplicates(['book_id', 'reader_id'], keep='first')
    to_delete = [el for el in df_dupl.index if el not in to_keep.index]
    return df.drop(to_delete)

def get_recommendation_by_user_ratings(vh: pd.DataFrame,
                                       book_id: int,
                                       recommendations_limit: int = 10
                                       ) -> pd.DataFrame:
    """
    In this approach properties of the books are discarded, only
    the user rating matters to make an estimate of the book
    similarity. We can still use book properties to filter the data
    Args:
        vh: matrix of relations on which we calculate cosine_similarity
        book_id: which book to compare
    Returns:

    """
    similarities = {}
    recommendations = []

    matching_books = [col for col in vh.columns if col != book_id]
    for col in matching_books:
        similarity = cosine_similarity(vh.loc[:, book_id], vh.loc[:, col])
        similarities[col] = similarity
    for i in range(recommendations_limit):
        best_match = max(similarities, key=similarities.get)
        recommendations.append(best_match)
        del similarities[best_match]

    print(f"Users who liked book {book_id} also liked {recommendations}")
    return recommendations


@cached(max_size=10)
def get_svd_decomposition(df):
    """
    Returns decomposition matrices, in which only one,
    relation between books and ratings is of particular interest
    """
    df = clean_duplicate_reviews(df)
    df_for_svg = df[['reader_id', 'book_id', 'book_rating']]
    matrix = df_for_svg.pivot(index="reader_id", columns="book_id",
                                 values="book_rating").fillna(0)

    u, s, vh = svd(matrix, full_matrices=False)
    return pd.DataFrame(vh, columns=matrix.columns)

path = 'Aufgabe_2_Testdaten.csv'

df = pd.read_csv(path, sep=",")
vh = get_svd_decomposition(df)
similar_books = get_recommendation_by_user_ratings(vh, book_id=5)

# For the case of genre filtering or language filtering:
vh = get_svd_decomposition(df[df['text_lang'] == 7])
similar_books = get_recommendation_by_user_ratings(vh, book_id=5)
