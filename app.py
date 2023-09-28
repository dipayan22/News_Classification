from flask import Flask,request,render_template
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# loading the model
MNB=pickle.load(open('model.pkl','rb'))

cv=pickle.load(open('cv.pkl','rb'))

# function for remove stopwords
def remove_stopwords(text):
    stopword=set(stopwords.words('english'))
    text=word_tokenize(text)
    
    return " ".join([x for x in text if x not in stopword])


# function to lemmatize the word
def lemmatize_word(text):
  wordnet = WordNetLemmatizer()
  return " ".join([wordnet.lemmatize(word) for word in text.split()])

# cv=CountVectorizer(max_features=5000)


app=Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':

        test=request.form['input_text']
        test=remove_stopwords(test)
        test=lemmatize_word(test)

        y_pred1 = cv.transform([test])

        yy =MNB.predict(y_pred1)

        

        return render_template('index.html',prediction_text="This news is basically written about : {}".format(yy[0]))


    return render_template("index.html")




if __name__=='__main__':
    app.run(debug=True)