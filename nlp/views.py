from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import pickle

category_data = pd.read_csv('/code/idx2category.csv')
idx2category = {row.k: row.v for idx, row in category_data.iterrows()}

with open('/code/model.pkl', 'rb') as file:
    model = pickle.load(file)


# Create your views here.
def index(request):
    if request.method == "GET": #アクセスされたら
        return render(
            request,
            'nlp/home.html'
        )
    else:
        title = request.POST.get("title", "")
        print('title:', title)
        result = model.predict([title])[0] #
        print('results:', result)
        pred = idx2category[result]
        #入力されたものを受け取る
        return render(
            request,
            "nlp/home.html",
            {"title" : pred} #値と変数をHtmlに送り込む
        )