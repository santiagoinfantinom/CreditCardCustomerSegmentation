import pandas as pd
import numpy as np
from sklearn import cluster
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

data = pd.read_csv('CC GENERAL.csv')
#print(data.head())

#Drop NA values
data = data.dropna()

#We will use BALANCE,PURCHASES,CREDIT_LIMIT to cluster different CC-Holders
clustering_data = data[['BALANCE','PURCHASES','CREDIT_LIMIT']]

#Import sklearn preprocesses' MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
for i in clustering_data.columns:
    MinMaxScaler(i) #Shrinks data from min -> 0 and max -> 1

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(clustering_data) #Create clusters for balance, purchases and credit limit
data['CREDIT_CARD_SEGMENTS'] = clusters

#For visualizing the 3 parts in a 3d graph
import plotly.graph_objects as go

PLOT = go.Figure()
for i in list(data["CREDIT_CARD_SEGMENTS"].unique()):
    PLOT.add_trace(go.Scatter3d(x=data[data["CREDIT_CARD_SEGMENTS"] == i]['BALANCE'],
                                y=data[data["CREDIT_CARD_SEGMENTS"] == i]['PURCHASES'],
                                z=data[data["CREDIT_CARD_SEGMENTS"] == i]['CREDIT_LIMIT'],
                                mode='markers', marker_size=6, marker_line_width=1,
                                name=str(i)))
PLOT.update_traces(hovertemplate='BALANCE: %{x} <br>PURCHASES %{y} <br>DCREDIT_LIMIT: %{z}')

PLOT.update_layout(width=800, height=800, autosize=True, showlegend=True,
                   scene=dict(xaxis=dict(title='BALANCE', titlefont_color='black'),
                              yaxis=dict(title='PURCHASES', titlefont_color='black'),
                              zaxis=dict(title='CREDIT_LIMIT', titlefont_color='black')),
                   font=dict(family="Gilroy", color='black', size=12))

PLOT.show() #Very very important. Otherwise, nothing gets shown