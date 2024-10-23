import pandas as pd

df_train = pd.read_pickle('train_final_RSA_class.pkl')
df_validation = pd.read_pickle('validation_final_RSA_class.pkl')
df_test = pd.read_pickle('test_final_RSA_class.pkl')

print ('TRAIN: ')
print (df_train[df_train['RSA_class'] == 0].shape[0])
print (df_train[df_train['RSA_class'] == 1].shape[0])
print (df_train[df_train['RSA_class'] == 2].shape[0])
print ('VALIDATION: ')
print (df_validation[df_validation['RSA_class'] == 0].shape[0])
print (df_validation[df_validation['RSA_class'] == 1].shape[0])
print (df_validation[df_validation['RSA_class'] == 2].shape[0])
print ('TEST: ')
print (df_test[df_test['RSA_class'] == 0].shape[0])
print (df_test[df_test['RSA_class'] == 1].shape[0])
print (df_test[df_test['RSA_class'] == 2].shape[0])