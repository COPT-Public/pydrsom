"""
a plot script for tensorboard download data,
e.g.
---------------------------------------
Wall time,Step,Value
---------------------------------------
1654943296.2150984,0,59.310001373291016
1654943313.7791796,1,64.77999877929688
1654943332.386608,2,70.02999877929688
1654943350.320845,3,71.08000183105469
1654943368.0967004,4,78.98999786376953
1654943386.335942,5,79.69999694824219
...
"""
import os
import sys

import plotly.graph_objects as go
import pandas as pd

NAME = {
  'adam': 'Adam',
  'rsomf': 'DRSOM',
  'sgd1': 'SGD-0.95',
  'sgd2': 'SGD-0.90',
  'sgd3': 'SGD-0.85',
  'sgd4': 'SGD-0.99',
  'rsomf-100': 'DRSOM-1e2',
  'rsomf-500': 'DRSOM-5e2',
  'rsomf-200': 'DRSOM-1e3',
  'adam-30': 'Adam-30',
  'adam-40': 'Adam-40',
  'drsom-mode:0-p': "DRSOM-2D",
  'drsom-mode:1-p': "DRSOM-2D-dg",
  'drsom-mode:2-p': "DRSOM-3D-dg",
  'drsom-mode:3-p': "DRSOM-2D-g-alone",
  'drsom-mode:3-p-r-20': "DRSOM-2D-g-alone-resume"
}
dfs = {}
dirin = sys.argv[1]
dirout = sys.argv[2]
for f in os.listdir(dirin):
  if not f.endswith('csv'):
    continue
  metric, cat, method = f.split('.')[-2].split("_")[0:3]
  metric = metric.lower()
  cat = cat.lower()
  method = method.lower()
  df = pd.read_csv(f"{sys.argv[1]}/{f}")
  df = df.rename(columns={k: k.lower() for k in df.columns})
  if df.shape[0] == 0:
    print(f'no data for {f}')
    continue
  # df['Wall time'] = pd.to_datetime(df['Wall time'])
  df['method'] = method
  df['subset'] = cat
  df[metric] = df['value']
  df = df.set_index(['subset', 'method', 'step']).drop(columns=['value'])
  if (cat, method) in dfs:
    dfs[cat, method][metric] = df[metric]
  else:
    dfs[cat, method] = df


def choose_color(m):
  if m == 'rsomf':
    return 'black'
  if m == 'rsomf-100':
    return 'rgb(128,128,128)'
  return 'rgb(60,60,60)'


# plots
# plot train
ranges = {
  'accuracy': [75, 102],
  'acc': [75, 102],
  'loss': [0, 1]
}
x_range = [0, 30]
# methods = ['adam', 'rsomf', 'sgd']
methods = NAME.keys()
for cat in ['train', 'test']:
  for metric in ['loss', 'acc']:
    data = [
      go.Line(x=dfs[cat, m].index.get_level_values(2),
              y=dfs[cat, m][metric],
              name=NAME.get(m, m),
              line=dict(width=2, color=choose_color(m)) if m in {'rsomf', 'rsomf-100', 'rsomf-200'} else dict(width=2)
              )
      for m in methods
      if (cat, m) in dfs
    ]
    opt_yaxis = dict(
      title=f"{metric}",
      color='black',
      range=ranges[metric]
    )
    if metric == 'loss':
      opt_yaxis['type'] = 'log'
      if cat == 'train':
        opt_yaxis['range'] = [-5, 1]
      else:
        opt_yaxis['range'] = [-1, 0]
    
    layout = go.Layout(
      plot_bgcolor='rgba(255, 255, 255, 1)',
      xaxis=dict(
        title=f"epoch",
        color='black',
        range=x_range
      ),
      yaxis=dict(opt_yaxis),
      font=dict(family="Latin Modern Roman", size=15),
      legend=dict(
        bordercolor='black',
        borderwidth=0.8,
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
      ),
    )
    fig = go.Figure(data=data, layout=layout)
    # update axis
    style_grid = dict(
      showline=True,
      linewidth=1.2,
      linecolor='black',
      showgrid=True,
      gridwidth=0.5,
      gridcolor='grey',
      griddash='dashdot',
    )
    fig.update_xaxes(style_grid)
    fig.update_yaxes(style_grid)
    fig.write_image(f"{dirout}/{cat}-{metric}.png", scale=10)
    fig.write_html(f"{dirout}/{cat}-{metric}.html")
