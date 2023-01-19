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
from tsmoothie.smoother import *

NAME = {
  'adam': 'Adam',
  'sgd1': 'SGDm-0.90',
  'sgd2': 'SGDm-0.95',
  'sgd3': 'SGDm-0.99',
  'drsom-g': 'DRSOM-g',
  'drsom-gd': 'DRSOM-g+d',
}
dfs = {}
dirin = sys.argv[1]
dirout = sys.argv[2]
for f in os.listdir(dirin):
  if f.endswith('csv'):
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
  if m == 'drsom-g':
    return 'black'
  if m == 'drsom-gd':
    return 'rgb(128,128,128)'
  return 'rgb(60,60,60)'


# plots
# plot train
ranges = {
  'accuracy': [75, 102],
  'acc': [0.7, 1.02],
  'loss': [0, 1]
}
# x_range = [0, 81]
x_range = [0, 30000]
methods = NAME.keys()
for cat in ['train', 'test']:
  # for metric in ['loss', 'acc']:
  for metric in ['loss', 'acc']:
    # for metric in ['acc']:
    data = []
    for m in methods:
      if (cat, m) not in dfs:
        continue
      ax = dfs[cat, m].index.get_level_values(2).to_list()
      if metric == 'loss':
        smoother = ExponentialSmoother(window_len=20, alpha=0.05)
      else:
        smoother = ExponentialSmoother(window_len=10, alpha=0.05)

      smoother.smooth(dfs[cat, m][metric])
      # low, up = smoother.get_intervals('prediction_interval', confidence=0.01)
      line = go.Line(
        x=ax,
        y=smoother.smooth_data[0],
        name=NAME.get(m, m),
        line=dict(width=1.5, color=choose_color(m))
        if m.startswith("drsom")
        else dict(width=1.5)
      )
      print(cat, metric, m, f"{smoother.smooth_data[0][-1]: .2f}")
      data.append(line)

    opt_yaxis = dict(
      title=f"{cat}-{metric}",
      color='black',
      range=ranges[metric]
    )
    if metric == 'loss':
      opt_yaxis['type'] = 'log'
      if cat == 'train':
        opt_yaxis['range'] = [-5, 1]
      else:
        opt_yaxis['range'] = [-1, 0]
    else:
      if cat == 'train':
        # opt_yaxis['range'] = [80, 102]
        opt_yaxis['range'] = [0.8, 1.02]
      else:
        # opt_yaxis['range'] = [80, 96]
        opt_yaxis['range'] = [0.80, 0.96]

    layout = go.Layout(
      plot_bgcolor='rgba(255, 255, 255, 1)',
      xaxis=dict(
        title=f"steps",
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
    fig.write_image(f"{dirout}/{cat}-{metric}.png", scale=3)
    fig.write_html(f"{dirout}/{cat}-{metric}.html")
    # break
