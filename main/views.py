from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from django.core.files.storage import FileSystemStorage
import catboost 
import joblib
from plotly.offline import plot
import plotly.graph_objects as go

model = joblib.load('model_catboost.pkl')

# Create your views here.
def base(request):

    return render(request, 'main/main.html')

def scoreJson(request):
    print(request.body)
    data=json.loads(request.body)
    df=pd.DataFrame({'x':data}).transpose()

    pred = model.predict(df)

    print(pred)

    return JsonResponse({'score': 1})

def scoreFile(request):
    FileObj=request.FILES
    File = FileObj['filePath']

    fs=FileSystemStorage()
    filePathName=fs.save(File.name, File)
    filePathName=fs.url(filePathName)
    filePath='.'+filePathName

    df_test = pd.read_csv(filePath)
    cat = request.POST['category']
    num = request.POST['numeric']
    categorical = df_test[cat].value_counts().sort_values(ascending=True)

    if request.method == 'POST' and request.POST['action'] == 'sum':

        numerical = df_test.groupby(cat).sum(num).reset_index()
        cat_list = numerical[cat].to_list()
        num_list = numerical[num].to_list()
        print(num_list)
        def bar():
            data = go.Bar(
                x=cat_list,
                y=num_list
            )

            layout = dict(
                title="Query Bot",
                yaxis=dict(range=[0,10000000])
            )

            fig = go.Figure(data=[data], layout=layout)
            plot_div = plot(fig, output_type='div', include_plotlyjs=False)
            return plot_div    

        context = {
                    'plot': bar()
                }
   
    return render (request, 'main/main.html', context)
    
  