import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df=pd.read_csv(r'A:\news.csv')
print("Shape of the dataframe:")
print(df.shape)
print("Head of the dataframe:")
print(df.head())
classify=df.label
x_train,x_test,y_train,y_test=train_test_split(df['text'], classify, test_size=0.2, random_state=7)
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
pac=PassiveAggressiveClassifier(max_iter=50,tol=0.001)
pac.fit(tfidf_train,y_train)
custom_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,custom_pred)
print(f'The accuracy is {round(score*100,3)}%')
print(confusion_matrix(y_test,custom_pred, labels=['FAKE','REAL']))