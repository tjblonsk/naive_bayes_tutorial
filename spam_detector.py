import pandas as pd

df = pd.read_table('./data/SMSSpamCollection', '\t', header=None, names=['label', 'sms_message'])

df.head()

df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
df.head() # returns (rows, columns)