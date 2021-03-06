
import sys
import pandas as pd
import sqlite3


def load_data(messages_filepath, categories_filepath):
    '''
    Reads data from messages and categories file and then outputs a merged dataframe.


    Args:
    messages_filepath : messages file.
    categories_filepath : categories file.

    Returns:
    df : merged dataframe.
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    '''

    Reads in the merged dataframe and prepares it for ML model.

    Args:
    df : dataframe.

    Returns:
    df : dataframe prepared for ML model.
    '''


    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    assert len(df[df.duplicated()]) == 0
    return df


def save_data(df, database_filename):

    '''

    Saves the pandas dataframe to a sqlite Database.

    Args:
    df : dataframe.
    database_filename : name of database.

    Returns:
    none
    '''


    conn = sqlite3.connect(database_filename)
    table_name = 'disaster'
    df.to_sql(table_name, con= conn, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        # print(df.shape)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()