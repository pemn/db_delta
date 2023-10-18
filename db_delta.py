#!python
# compare two datasets into a change matrix
# key_a, key_b: (optional) primary key(s) to relate records between datasets
# group_a, group_b: the classification variable to observe changes (ex.: lito)
# value_a, value_b: (optional) a numeric variable with the quantity of change (ex.: volume)
# condition: (optional) python expression restricting records to be used
# output_matrix: (optional) path to save the migration table
# output_records: (optional) path to save the side by side records
# v3.0 2023/10 paulo.ernesto
# v2.0 2021/04 paulo.ernesto
# v1.0 2020/09 paulo.ernesto
'''
usage: $0 input_a*csv,xlsx,bmf,isis keys_a#key:input_a groups_a#group_a:input_a value_a:input_a input_b*csv,xlsx,bmf,isis keys_b#key:input_b groups_b#group_b:input_b value_b:input_b condition output_matrix*xlsx percent@ output_records*csv
'''
import sys, os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, pd_load_dataframe, log

def fn_eye_style(d, s = 'background-color: lightgray'):
  return pd.DataFrame(np.where(np.eye(d.shape[0], d.shape[1]), s, ''), index=d.index, columns=d.columns)

def pd_diff_bar_2d(df, page_size = None):
  if page_size is not None:
    page_size = max(1, int(page_size))
    if len(df) > page_size:
      bs = np.argsort(np.sum(df, 1))
      df = df.iloc[bs[-page_size:]]
  # create a grid on the output image that fits the data
  nrows = int(np.sqrt(len(df)))
  ncols = int(np.ceil(len(df) / nrows))
  fig, ax = plt.subplots(nrows, ncols, tight_layout=True, sharex=True, sharey=True, squeeze=False, figsize=np.multiply(plt.rcParams["figure.figsize"], 2))
  i = 0
  for ri,rd in df.iterrows():
    #yd = rd - df[ri]
    yd = df[ri] - rd
    if i > ax.size:
      break
    ax.flat[i].bar(np.arange(len(rd)), yd, tick_label = rd.index)
    ax.flat[i].set_title(ri)
    i += 1

  plt.show()

def pd_diff_bar_3d(df):
  z = 0
  fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
  plt.set_cmap('Spectral')
  cmap = plt.get_cmap()
  for ri,rd in df.iterrows():
    #yd = rd - df[ri]
    yd = df[ri] - rd
    ax.bar(np.arange(len(rd)), yd, zs=z, zdir='y', color=cmap(z / len(df)), label=ri)
    z += 1

  ax.set_xticks(np.arange(df.shape[0]))
  ax.set_xticklabels(df.columns)
  ax.set_yticks(np.arange(df.shape[1]))
  ax.set_yticklabels(df.index)
  fig.legend()
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

def pd_major_sum(_):
  r = np.nan
  if _.dtype.num == 17:
    vc = _.value_counts()
    if vc.size:
      r = vc.idxmax()
  else:
    r = np.nansum(_)
  return r

def pd_ijk_merge_delta(il, condition):
  dfv = []
  dfg = []
  dfa = None
  for i in range(len(il)):
    fn,groups,key,value = il[i]
    gl = groups.split(';')
    bn = os.path.splitext(os.path.basename(fn))[0]
    vi = '%s:%s:%d' % (bn, value, i)
    dfv.append(vi)
    gs = ['%s:%s:%d' % (bn,_,i) for _ in gl]
    dfg.append(gs)
    kl = None
    if key:
      kl = key.split(';')
    df = pd_load_dataframe(fn, condition, None, [value] + kl + gl)
    if not value:
      value = vi
      df[value] = 1
    df.rename(columns=dict(((value, vi),) + tuple(zip(gl, gs))), inplace=True)
    if kl is not None:
      df = df.pivot_table([vi] + gs, kl, None, pd_major_sum)

    if sys.hexversion < 0x3080000:
      if df.ndim == 1:
        # pd.merge in older versions only works on dataframes
        df = df.to_frame()
      df.reset_index(inplace=True)
      #df.to_excel('merge_%d.xlsx' % i)
      if dfa is None:
        dfa = df
      else:
        dfa = pd.merge(dfa, df, 'outer', kl)
    else:
      if dfa is None:
        dfa = df
      else:
        dfa = pd.merge(dfa, df, 'outer', left_index=True, right_index=True)

  if sys.hexversion >= 0x3080000:
    dfa.reset_index(inplace=True)
  # remove nans lest they crash sort_values
  for gl in dfg:
    for g in gl:
      dfa[g].fillna('', inplace=True)

  return dfa, dfg, dfv

def pd_delta(dfa, dfg, dfv):
  log('index')
  aks = None
  for d in dfg:
    r = pd.Index(dfa[d].drop_duplicates())
    if aks is None:
      aks = r
    else:
      aks = aks.union(r)

  aks = aks.sort_values()
  
  log('repivot')
  df = pd.DataFrame(np.zeros((len(aks), len(aks))), aks.copy(','.join(dfg[0])), aks.copy(','.join(dfg[1])))

  for ri, rd in dfa.iterrows():
    if ri % 10000 == 0:
      log(ri,'/',dfa.shape[0],'records processed')
    g0 = tuple(rd[dfg[0]])
    g1 = tuple(rd[dfg[1]])
    v0 = rd[dfv[0]]
    v1 = rd[dfv[1]]
    #   df.loc[g0, g1] += min(v0, v1)
    #   df.loc[g0, g0] += max(0, v0 - v1)
    #   df.loc[g1, g1] += max(0, v1 - v0)
    # .loc not working for some reason
    df[g1][g0] += min(v0, v1)
    df[g0][g0] += max(0, v0 - v1)
    df[g1][g1] += max(0, v1 - v0)

  if ri < dfa.shape[0]:
    log(dfa.shape[0],'/',dfa.shape[0],'records processed')

  return df


def db_delta(input_a, keys_a, groups_a, value_a, input_b, keys_b, groups_b, value_b, condition, output_matrix, percent = False, output_records = None):
  il = [[input_a, groups_a, keys_a, value_a],[input_b, groups_b, keys_b, value_b]]

  log('merge')
  dfa, dfg, dfv = pd_ijk_merge_delta(il, condition)
  df = pd_delta(dfa, dfg, dfv)

  if int(percent):
    df /= df.sum()

  #df.columns = pd.MultiIndex.from_tuples([('', ' '.join(_)) for _ in df.columns], name=[df.columns.name,''])
  df.index.values[:] = np.array(df.index.map(lambda _: ' '.join(_)))

  log('output matrix')
  if output_matrix:
    df.to_excel(output_matrix, index=True)
    #df.style.apply(fn_eye_style, None).to_excel(output_matrix)
  else:
    print(df.to_string())

  if output_records:
    log('output records')
    dfa.to_csv(output_records, index=False)

  log("finished")

main = db_delta

if __name__=="__main__":
  usage_gui(__doc__)
