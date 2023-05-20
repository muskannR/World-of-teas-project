from django.shortcuts import render, redirect, \
 reverse, HttpResponse
from products.models import Product
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# Create your views here.

def recommend(request,title):

    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404


    df=pd.DataFrame(list(Products.objects.filter(has_weight=True)))

    
    # Using Abhishek Thakur's arguments for TF-IDF
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

    # Filling NaNs with empty string
    df['description'] = df['description'].fillna('') 
    tfv_matrix = tfv.fit_transform(df['description'])


    # Compute the sigmoid kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

    # Reverse mapping of indices and movie titles
    indices = pd.Series(df.index, index=df['name']).drop_duplicates()

    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the  
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar 
    sig_scores = sig_scores[1:11]

    #  indices
    tea_indices = [i[0] for i in sig_scores]

    

    tea_list = list(df['name'].iloc[tea_indices])
    context = {'tea_list': tea_list}
    return render(request, 'recommend/recommend.html', context)
