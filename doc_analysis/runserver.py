from models.analysis import Result
from forms import  ChooseId, TargetWord

import os
from flask import Flask, render_template, url_for, redirect, request
from werkzeug import secure_filename
import time
import pickle
import pandas as pd

model_result = Result()

app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload',methods = ['GET','POST'])
def upload():

    if request.method =='POST':

        files = request.files.getlist('file')
        if files:

            for f in files:
                basepath = os.path.dirname(__file__)
                upload_path = os.path.join(basepath,'static/uploads', secure_filename(f.filename))
                f.save(upload_path)

            file_path = os.getcwd()
            doc_path = file_path + '/static/uploads/'

            raw_dict, result, voca = model_result.get_data(doc_path)

            freq = voca.sort_values('word_frequency', ascending=False).iloc[:10,:]

            result.to_pickle('static/result/result.pkl')

            with open('static/result/raw_dict.pkl', 'wb') as file:
                pickle.dump(raw_dict, file)


            return render_template('success.html', freq=freq.to_html())
    return render_template('upload.html')



@app.route('/folder',methods = ['GET','POST'])
def folder():

    form = TargetWord()
    if request.method =='POST':

        result = pd.read_pickle('static/result/result.pkl')

        word_df = result[result.word==form.targetword.data]

        return render_template('folder_result.html',word_df=word_df.to_html())
    return render_template('folder.html',form=form)


@app.route('/document',methods = ['GET','POST'])
def doc():

    form = ChooseId()
    id = form.id.data
    if request.method =='POST':
        result = pd.read_pickle('static/result/result.pkl')

        with open('static/result/raw_dict.pkl', 'rb') as f:
            raw_dict = pickle.load(f)

        wc_image, keywords, keyphrases, topics, corr = model_result.nlp_result(id, result, raw_dict)
        return render_template('doc_result.html',
                               id = id,
                               keywords=keywords,
                               keyphrases=keyphrases,
                               topics=topics,
                               val1=time.time())
    return render_template('doc.html',form=form)


@app.route('/result')
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
