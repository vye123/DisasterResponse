# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

# load messages dataset
def load_data(messages_filepath, categories_filepath):
    
    """    
    Input:
        messages_file -> messages CSV file 
        categories_file -> categories CSV file
    Output:
        df -> a merged dataframe that includes all messages and categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge data sets
    df = messages.merge(categories,on='id')
    return df 



def clean_data(df):
    
    """
    Clean data to prepare for building the ML model later on
    
    Input:
        df -> The merged data frame from load_file function
    Output:
        df -> A cleaned data frame ready to be saved to Database
    """
    
    # Split the categories
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:].values

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [r[:-2] for r in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df[categories.columns] = categories
    
    # drop duplicates
    df.drop_duplicates()
    
    return df



def save_data(df, database_filename):
    
    """
    Save cleaned data frame to SQLite Database table
    
    Input:
        df -> The cleaned data frame from clean_data function
        database_filename -> Database name
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')
  

def main():
    
    """
    Main function which will kick off the data processing functions. There are three primary actions taken by this function:
        1) Load Messages Data with Categories
        2) Clean Categories Data
        3) Save Data to SQLite Database
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()