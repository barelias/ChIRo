import pandas as pd

def one_hot_to_class(one_hot: str):
    if one_hot == [1, 0, 0]: return 0
    if one_hot == [0, 1, 0]: return 1
    if one_hot == [0, 0, 1]: return 2
    raise Exception

test_final_RSA = pd.read_pickle("test_final_RSA.pkl")
train_final_RSA = pd.read_pickle("train_final_RSA.pkl")
validation_final_RSA = pd.read_pickle("validation_final_RSA.pkl")

test_final_RSA['RSA_class'] = test_final_RSA['RSA_label_one_hot'].apply(one_hot_to_class)
train_final_RSA['RSA_class'] = train_final_RSA['RSA_label_one_hot'].apply(one_hot_to_class)
validation_final_RSA['RSA_class'] = validation_final_RSA['RSA_label_one_hot'].apply(one_hot_to_class)

test_final_RSA.to_pickle('test_final_RSA_class.pkl')
train_final_RSA.to_pickle('train_final_RSA_class.pkl')
validation_final_RSA.to_pickle('validation_final_RSA_class.pkl')