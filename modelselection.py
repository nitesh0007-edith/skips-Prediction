import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle


#Loaidng the datasets
df = pd.read_csv('technocolabs training set.csv',low_memory=False)
feat = pd.read_csv('tf_000000000000.csv',low_memory=False)

df.rename(columns = {'track_id_clean':'track_id'},inplace=True)

main = pd.merge(df,feat)

main.head(5)

#Droping the features which wont be useful for further analysis
main.drop(['session_id','track_id','date','skip_1','skip_2','skip_3'],axis=1,inplace=True)

#Categorical data encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
notskipped = le.fit_transform(main['not_skipped'])
premium = le.fit_transform(main['premium'])
contexttype = le.fit_transform(main['context_type'])
histstart = le.fit_transform(main['hist_user_behavior_reason_start'])
histend = le.fit_transform(main['hist_user_behavior_reason_end'])
shuffle = le.fit_transform(main['hist_user_behavior_is_shuffle'])
mode = le.fit_transform(main['mode'])

#Dropping the encoded features
main.drop(['not_skipped','premium','context_type','hist_user_behavior_reason_start','hist_user_behavior_reason_end','hist_user_behavior_is_shuffle','mode'],axis = 1,inplace=True)

#Covetring arrays to Series
premium = pd.Series(premium,name='premium')
contexttype = pd.Series(contexttype,name='context_type')
histstart = pd.Series(histstart,name='hist_user_behavior_reason_start')
histend = pd.Series(histend,name='hist_user_behavior_reason_end')
notskipped = pd.Series(notskipped,name='not_skipped')
shuffle = pd.Series(shuffle,name='hist_user_behavior_is_shuffle')
mode = pd.Series(mode,name='mode')

#Concating the dataframe and the encoded data
main = pd.concat([main,premium,contexttype,histstart,histend,shuffle,mode,notskipped],axis=1)

X = main[['session_position','session_length','no_pause_before_play','short_pause_before_play','long_pause_before_play','hist_user_behavior_n_seekfwd','hist_user_behavior_n_seekback','hour_of_day','catalog', 'charts', 'editorial_playlist', 'personalized_playlist',
       'radio', 'user_collection', 'appload', 'backbtn', 'clickrow', 'endplay',
       'fwdbtn', 'playbtn', 'remote', 'trackdone', 'trackerror']]

y =main['not_skipped']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
predictrfc = rfc.predict(X_test)

pickle.dump(rfc,open('skip_prediction.pkl','wb'))
