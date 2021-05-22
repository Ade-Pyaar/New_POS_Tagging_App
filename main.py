from flask import Flask, render_template, request
from decouple import config
from utils import viterbi_backward, get_emission_and_vocab, my_preprocess, predict_pos
from pandas import read_csv



app = Flask(__name__)
app.config["SECRET_KEY"] = config('SECRET_KEY')
app.config['JSON_SORT_KEYS'] = False



@app.route("/", methods=["GET", "POST"])
def index():
    suggestion = {}

    my_csv = read_csv('tags.csv')

    result = {x:y for x, y in zip(my_csv['Tags'], my_csv['Meaning'])}


    if request.method == "POST":
        my_text = request.form.get("word").strip().lower()
        model_to_use = request.form.get('model').strip().lower()


        if 'simple' in model_to_use:
            vocab, emission_count = get_emission_and_vocab()
            orig, prep = my_preprocess(my_text)
            final = predict_pos(prep, emission_count, vocab)
        
        else:
            orig, prep = my_preprocess(my_text)
            final = viterbi_backward(prep)

        for i in range(len(orig)):
            suggestion[orig[i]] = final[i]
        
        print(suggestion)

    return render_template("index.html", suggestion=suggestion, result=result)