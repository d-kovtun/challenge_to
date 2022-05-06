import pandas as pd
import numpy as np
from scipy.linalg import sqrtm


def replace_nan_with_mean(df):
    """
    Means inserted by columns instead of nan values
    Args:
        df: dataframe with numerical values only

    Returns:
        Masked Array, which is the input dataframe, but nan replaced with
        means
    """
    df = np.array(df, dtype=float)

    mask = pd.isna(df)
    masked_arr = np.ma.masked_array(df, mask)
    col_means = np.mean(masked_arr, axis=0)
    df = masked_arr.filled(col_means)

    return df, col_means

def get_reduced_svd_decomposition(df, k):
    """
    Subtract average from the utility matrix to work with deltas.
    Relation between users and ratings in u,
    between books and ratings in vh
    Args:
        df: pivot matrix of the user-book-rating values
        k: reduced rank of the decomposition
    Returns:
        Decomposition matrix
    """
    utility_matrix, col_means = replace_nan_with_mean(df)
    avg = np.tile(col_means, (utility_matrix.shape[0], 1))
    utility_matrix = utility_matrix - avg

    u, s, vh = np.linalg.svd(utility_matrix, full_matrices=False)
    s = np.diag(s)

    # Only k most significant features
    s = s[0:k, 0:k]
    u = u[:, 0:k]
    vh = vh[0:k, :]

    s_root = sqrtm(s)

    # U * S * S * V
    usk = np.dot(u, s_root)
    skvh = np.dot(s_root, vh)
    usvh = np.dot(usk, skvh)

    usvh = usvh + avg
    return usvh


def rmse(real, forecast):
    diff = real - forecast
    return sum([sigma**2 for sigma in diff]) / len(diff)


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


def get_data(path='Aufgabe_2_Testdaten.csv'):
    data = pd.read_csv(path, sep=',')
    data['reader_id'] = data['reader_id'].astype('str')
    data['book_id'] = data['book_id'].astype('str')
    return data


def split_data_into_sets(df, test_to_train_ratio = 0.2):
    """
    Splits into test and training set

    """
    readers = df['reader_id'].unique()

    test_index = []
    train_index = []
    for i, u in enumerate(readers):
        temp = df[df['reader_id'] == u]
        n = len(temp)
        test_size = int(test_to_train_ratio * n)

        # dummy_test = temp.iloc[n - 1 - test_size:]
        # dummy_train = temp.iloc[: n - 2 - test_size]
        test_index += list(temp.iloc[n - 1 - test_size:].index)
        train_index += list(temp.iloc[: n - 2 - test_size].index)
        # test = pd.concat([test, dummy_test])
        # train = pd.concat([train, dummy_train])
        if i % 1000 == 0:
            print(f"Split {i}/{len(readers)} readers into the test "
                  f"and training set")
    test = df.loc[test_index, :]
    train = df.loc[train_index, :]
    return test, train


def tweak_ranking(pivot_matrix, reader_index, books_index,
                  test_df,
                  possible_ranks=[5, 10]):

    best_rank = possible_ranks[0]
    best_rmse = 999
    for f in possible_ranks:
        svdout = get_reduced_svd_decomposition(pivot_matrix, k=f)
        pred = []
        rows_index = []
        for i, row in test_df.iterrows():
            user = row['reader_id']
            book = row['book_id']
            try:
                u_index = reader_index[user]
                rows_index.append(i)
            except KeyError:
                # ToDo: insure that the cases of a reader being only in training
                # or only testing case are excluded beforehand
                continue
            if book in books_index:
                i_index = books_index[book]
                pred_rating = svdout[u_index, i_index]
            else:
                pred_rating = np.mean(svdout[u_index, :])
            pred.append(pred_rating)

        rmse_res = rmse(test_df.loc[rows_index, 'book_rating'], pred)
        if rmse_res < best_rmse:
            best_rank = f
            best_rmse = rmse_res
        print(f"Done checking rank {f} on test set. ")
    return best_rank, svdout


def test_train_to_get_rank(df):
    df = clean_duplicate_reviews(df)

    # First, train the model to guess the ranking for the SVD
    test, train = split_data_into_sets(df)

    train.reset_index(inplace=True, drop=True)
    pivot_matrix = train.pivot(index="reader_id", columns="book_id",
                               values="book_rating")
    reader_index = {val: i for i, val in enumerate(pivot_matrix.index)}
    books_index = {val: i for i, val in enumerate(pivot_matrix.columns)}

    rank, _ = tweak_ranking(pivot_matrix, reader_index, books_index, test)
    return rank

if __name__ == "__main__":
    df = get_data('Aufgabe_2_Testdaten.csv')
    df = clean_duplicate_reviews(df)
    # Optional, slow for the full dataset. Set `rank` to any other value,
    # e.g. rank = 10
    rank = test_train_to_get_rank(df)

    # Now real prediction with the calculated rank.
    # Plug any other user_id, since the matrix is calculated for all users
    # ToDo: should exclude those already rated by the reader
    pivot_matrix = df.pivot(index="reader_id", columns="book_id",
                            values="book_rating")
    reader_index = {val: i for i, val in enumerate(pivot_matrix.index)}
    books_index = {val: i for i, val in enumerate(pivot_matrix.columns)}
    svdout = get_reduced_svd_decomposition(pivot_matrix, k=rank)

    user_id = 1
    all_user_ratings = svdout[reader_index[str(user_id)], :]
    top_recommendations = sorted(range(len(all_user_ratings)),
                                    key=lambda i: all_user_ratings[i])[-10:]
    list_of_values = list(books_index.values())
    recommended_book_ids = []
    for recommend_index in top_recommendations:
        book_id = books_index[str(list(books_index.values()
                                       ).index(recommend_index))]
        recommended_book_ids.append(book_id)
    print(f"We recommend user {user_id} books {recommended_book_ids}")

