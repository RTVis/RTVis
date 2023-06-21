from collections import Counter
import pandas as pd

stopwords = pd.read_csv('./assets/data/stop_words.csv')
sw = stopwords['0'].tolist()
sw.append('0')

df = pd.read_excel('./demo_dataset.xlsx')
dframe = df[['Date','Abstract']]
dframe = dframe.dropna()
dframe['Abstract'] = dframe['Abstract'].apply(lambda x: ''.join([i.lower() for i in x if i.isalpha() or i == " "]) if x != None else x)\
                                      .apply(lambda x: ' '.join([i for i in x.split() if i not in sw]) if x != None else x)\
                                      .apply(lambda x: ' '.join(list(set(x.split()))) if x != None else x)
                                      
#for each row, append to a string str
str = ''
for index, row in dframe.iterrows():
    str = str + row['Abstract']
    
#counter the words
c = Counter(str.split())
df3 = pd.DataFrame.from_dict(c, orient='index').reset_index()
df3.columns = ['word', 'count']
df3 = df3.sort_values(by=['count'], ascending=False)
df3 = df3.head(500)
df3 = df3.reset_index(drop=True)

#make a dictionary with key as words and value as 0
d = dict.fromkeys(df3['word'].tolist(), 0)

#group by date
df3 = dframe.groupby('Date')
df3 = df3.apply(lambda x: x.sort_values(['Date'], ascending=[True]))
#delete the "Date" column
df3 = df3.reset_index(drop=True)

#connect all abstracts in the same date to one string
df4 = df3.groupby('Date')['Abstract'].apply(lambda x: "%s" % ' '.join(x))
df4 = pd.DataFrame(df4)
#reset index
df4 = df4.reset_index()
df4.head(10)

#new df to create barcharrace
df5 = pd.DataFrame(columns=['Date', 'Word', 'Count'])

#for each abstract, call remove function to remove stopwords and punctuations
for index, row in df4.iterrows():
    #for each word in the abstract, add 1 to the dictionary
    for word in row['Abstract'].split():
        #if wor in the dictionary, add 1
        if word in d:
            d[word] += 1
    #sort the dictionary by value
    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    #get the first 15 words and save to the dataframe use concat
    for i in range(15):
        df5 = pd.concat([df5, pd.DataFrame([[row['Date'], list(d.keys())[i], list(d.values())[i]]], columns=['Date', 'Word', 'Count'])])

#save to csv
df5.to_csv('./assets/data/barChartRace.csv', index=False)