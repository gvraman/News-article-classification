from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier

import nltk
from nltk.corpus import stopwords

import numpy as np
from sklearn.model_selection import train_test_split

# load the model from disk
filename = 'xgb_model_57.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform_new.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
#    df1 = pd.read_csv("articles1.csv")
#    df2 = pd.read_csv("articles2.csv")
#    df3 = pd.read_csv("articles3.csv")
#    df4 = pd.concat([df1, df2, df3], ignore_index=True)
#    df = df4.drop(['Unnamed: 0','id', 'title', 'author', 'date', 'year', 'month', 'url'], 1)
#    df['publication'] = df['publication'].map({'Breitbart': 0, 'New York Post': 1, 'NPR': 2, 'CNN': 3, 'Washington Post': 4, 'Reuters': 5, 'Guardian': 6, 'New York Times': 7,'Atlantic': 8,'Business Insider': 9,'National Review': 10,'Talking Points Memo': 11,'Vox': 12,'Buzzfeed News': 13,'Fox News': 14})\
#    #df = df.drop(['publication'],1)
#    #	X = df['message']
#    #	y = df['label']
#    
#    
#    df['content'] = df['content'].str.split()
#    nltk.download('stopwords')
#    stop = stopwords.words('english')
#    df['content'] = df['content'].apply(lambda x: [word for word in x if word not in stop])
#    df['content'] = df['content'].apply(' '.join)
#    
#    #df1 = pd.concat([Atlantic,Breitbart,BusinessInsider,NewYorkTimes], axis = 0 )
#    # print(df1.shape)
#    X = df.iloc[:, 1].values #Content of articles
#    Y = df.iloc[:, 0].values #Label
#    
#    from sklearn.feature_extraction.text import TfidfVectorizer 
#    cv = TfidfVectorizer(max_df = 0.5,min_df = 0.1).fit(X)
#    X = cv.transform(X)
#    
#    pickle.dump(cv, open('tranform_new.pkl', 'wb'))
#    
#    import numpy as np
#    #np.shape(X)
#    #print(X)
#    
#    from sklearn.model_selection import train_test_split
#    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
#    
#    #MLPClassifier
#    clf = MLPClassifier(activation = 'tanh',hidden_layer_sizes=(15,), random_state = 1)
#    clf.fit(X_train,y_train)
#    pred=clf.predict(X_test)
#    print(accuracy_score(pred,y_test))
#    
#    clf = xgb.XGBClassifier()
#    clf.fit(X_train,y_train)
#    pred=clf.predict(X_test)
#    print(accuracy_score(pred,y_test))
#    
#    clf = DecisionTreeClassifier()
#    clf.fit(X_train,y_train)
#    pred=clf.predict(X_test)
#    print(accuracy_score(pred,y_test))
#    
#    filename = 'Dtree_40.pkl'
#    pickle.dump(clf, open(filename, 'wb'))
#    
#    
#    
#    
#    #Output:
#    '''
#    Out[24]:
#    MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
#    beta_2=0.999, early_stopping=False, epsilon=1e-08,
#    hidden_layer_sizes=(15,), learning_rate='constant',
#    learning_rate_init=0.001, max_iter=200, momentum=0.9,
#    nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#    solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
#    warm_start=False)
#
#    0.7570630979434952
#    '''
#
#    #KNN
#    '''
#    clf=KNeighborsClassifier()
#    clf.fit(X_train,y_train)
#    pred=clf.predict(X_test)
#    print (accuracy_score(pred,y_test))
#    print(clf)
#    '''
#    #Naive Bayes
#    '''
#    clf = MultinomialNB(alpha=0.5)
#    clf.fit(X_train, y_train)
#    pred=clf.predict(X_test)
#    print (accuracy_score(pred,y_test))
#    '''
#    
#    #SVM with Grid Search
#    '''
#    clf = svm.SVC()
#    Cs = [0.001, 0.01, 0.1, 1, 10]
#    gammas = [0.001, 0.01, 0.1, 1]
#    param_grid = {'C': Cs, 'gamma' : gammas}
#    grid_search = GridSearchCV(clf, param_grid, cv=2)
#    grid_search.fit(X_train,y_train)
#    pred=clf.predict(X_test)
#    print (accuracy_score(pred,y_test))
#    '''
#    #Random Forest
#    '''
#    clf = RandomForestClassifier()
#    param_grid = {'bootstrap': [True],'max_depth': [8, 9],'max_features': [2, 3],'min_samples_leaf': [4, 5],'min_samples_split': [8, 10],'n_estimators': [100, 200]}
#    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 1)
#    grid_search.fit(X_train, y_train)
#    grid_search.best_params_
#    pred=grid_search.predict(X_test)
#    print(accuracy_score(pred,y_test))
#    '''
#    
#    #clf = DecisionTreeClassifier()
#    #clf = RandomForestClassifier(n_estimators=50, max_depth = None, min_samples_split=5, random_state = 0)
#    #clf = svm.SVC()
#    #clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
#    #clf = xgb.XGBClassifier()
#    #clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
#    #clf = SGDClassifier()
#    #clf = KNeighborsClassifier(n_neighbors=30)
#    #clf = GradientBoostingClassifier(n_estimators = 320)
    
    
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)