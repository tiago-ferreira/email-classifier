import pandas as pd

def readEmail():
    df = pd.read_csv('emails.csv', encoding = 'utf-8')
    X_df = df[['email']]
    Y_df = df['classificacao']

    Xdummies_df = pd.get_dummies(X_df)
    Ydummies_df = Y_df

    X = Xdummies_df.values
    Y = Ydummies_df.values
    return X,Y
