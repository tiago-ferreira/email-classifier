import pandas as pd

def readEmail():
    df = pd.read_csv('emails.csv', encoding = 'utf-8')
    X = df['email']
    Y = df['classificacao']
    return X,Y
