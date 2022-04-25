#!python
# compare two datasets into a change matrix
# key_a, key_b: (optional) primary keys (comma separated) to relate records between datasets
# group_a, group_b: the classification variable to observe changes
# value_a, value_b: (optional) a numeric variable to as change quantity
# condition: (optional) python expression restricting records to be used
# output_matrix: (optional) path to save the migration table
# output_records: (optional) path to save the side by side records
# chart_bar: (optional) display 3d bar charts showing relative mass changes
# chart_scatter: (optional) display 3d scatter showing the regions that changed
# v2.0 04/2021 paulo.ernesto
# v1.0 09/2020 paulo.ernesto
'''
usage: $0 input_a*csv,xlsx,bmf,isis key_a:input_a group_a:input_a value_a:input_a input_b*csv,xlsx,bmf,isis key_b:input_b group_b:input_b value_b:input_b condition output_matrix*xlsx output_records*xlsx chart_bar@ chart_scatter@3 x:input_a y:input_a z:input_a
'''
import sys, os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# import modules from a pyz (zip) file with same name as scripts
sys.path.append(os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, pd_load_dataframe, pd_save_dataframe

def fn_eye_style(d, s = 'background-color: lightgray'):
  return pd.DataFrame(np.where(np.eye(d.shape[0], d.shape[1]), s, ''), index=d.index, columns=d.columns)

def pd_diff_bar_3d(df):
  z = 0
  fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
  plt.set_cmap('Paired')
  cmap = plt.get_cmap()
  for ri,rd in df.iterrows():
    yd = rd - df[ri]
    ax.bar(np.arange(len(rd)), yd, zs=z, zdir='y', color=cmap(z / len(df)))
    z += 1
  ax.set_xticklabels(df.index)
  ax.set_yticklabels(df.columns)
  plt.show()

def pd_diff_scatter_3d(xc, yc, zc, s1, s2):
  bi = np.asarray(s1) != np.asarray(s2)
  arr = ['%s➔%s' % (t1, t2) for t0, t1, t2 in zip(bi, s1, s2) if t0]
  if len(arr):
    c,l = pd.factorize(arr)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    cmap = plt.set_cmap('Paired')
    scat = plt.scatter(xc[bi], yc[bi], 100, zs = zc[bi], marker=".", c=c)
    #fig.colorbar(scat)
    cmap = plt.get_cmap()
    fig.legend([mpatches.Patch(color=cmap(_ / max(1,(len(l)-1)))) for _ in range(len(l))], l)
  else:
    plt.figtext(0.4, 0.5, "✔ all data is equal")
  plt.show()

def main(input_a, key_a, group_a, value_a, input_b, key_b, group_b, value_b, condition, output_matrix, output_records = None, chart_bar = None, chart_scatter = None, x = None, y = None, z = None):

  df_a = pd_load_dataframe(input_a, condition, None, [key_a, group_a, value_a])
  df_b = pd_load_dataframe(input_b, condition, None, [key_b, group_b, value_b])
  df_a.fillna('null', inplace=True)
  df_b.fillna('null', inplace=True)

  if key_a:
    df_a.set_index(key_a.split(','), drop=False, inplace=True)
  if key_b:
    df_b.set_index(key_b.split(','), drop=False, inplace=True)

  axb = dict()
  aks = set()
  arl = list()
  g_a = None
  for i_a,row_a in df_a.iterrows():
    if group_a:
      g_a = row_a[group_a]
    aks.add(g_a)
    v_a = 1
    if value_a:
      v_a = row_a[value_a]
    g_b, v_b = None, None
    if i_a in df_b.index:
      row_b = df_b.loc[i_a]
      if np.ndim(row_b) == 2:
        row_b = row_b.max(0)
      if group_b:
        g_b = row_b[group_b]
      for k in [g_a, g_b]:
        if k not in axb:
          axb[k] = dict([(g_a, 0), (g_b, 0)])
        if g_b not in axb[k]:
          axb[k][g_b] = 0

      aks.add(g_b)
      v_b = 1
      if value_b:
        v_b = row_b[value_b]
      axb[g_a][g_b] += v_b - max(0, v_b - v_a)
      axb[g_a][g_a] += max(0, v_a - v_b)
      axb[g_b][g_b] += max(0, v_b - v_a)
    arl.append([i_a,g_a,v_a,g_b,v_b])

  aks = sorted(aks)

  name_a = "%s:%s" % (group_a, os.path.basename(input_a))
  name_b = "%s:%s" % (group_b, os.path.basename(input_b))
  df_axb = pd.DataFrame(np.zeros((len(aks), len(aks))), pd.MultiIndex.from_product([[name_a], aks]), pd.MultiIndex.from_product([[name_b], aks]))
  for row_a in df_axb.index:
    for row_b in df_axb.columns:
      df_axb.loc[row_a, row_b] = axb.get(row_a[1], {}).get(row_b[1], 0)

  if output_matrix:
    if sys.hexversion < 0x3060000:
      df_axb.to_excel(output_matrix)
    else:
      df_axb.style.apply(fn_eye_style, None).to_excel(output_matrix)
  else:
    print(df_axb.to_string())
  if output_records:
    df = pd.DataFrame(arl, columns=[key_a, name_a, value_a or 'a', name_b, value_b or 'b'])
    pd_save_dataframe(df, output_records)
  if int(chart_bar):
    df = None
    if sys.hexversion < 0x3060000:
      df = df_axb.copy()
      df.index = df.index.levels[1]
      df.columns = df.columns.levels[1]
    else:
      df = df_axb.droplevel(0).droplevel(0, 1)
    pd_diff_bar_3d(df)

  if int(chart_scatter):
    pd_diff_scatter_3d(df_a[x], df_a[y], df_a[z], [_[1] for _ in arl], [_[3] for _ in arl])

  print("finished")


if __name__=="__main__":
  usage_gui(__doc__)
