#!python
# create a custom waterfall chart with changes between two block models
# key_a, key_b: (optional) primary key(s) to relate records between datasets
# groups_a, groups_b: the classification variable(s) to observe changes (ex.: lito)
# value_a, value_b: (optional) a numeric variable with the quantity of change (ex.: volume)
# mode: generate either a waterfall chart or a custom bar comparison chart
# more detail on manual
# v1.0 2023/08 paulo.ernesto
'''
usage: $0 input_a*vtk,csv,xlsx,bmf,isis keys_a#key:input_a groups_a#group_a:input_a value_a:input_a input_b*vtk,csv,xlsx,bmf,isis keys_b#key:input_b groups_b#group_b:input_b value_b:input_b condition mode%waterfall,compare,all,none output_table*xlsx output_chart*pdf display@
'''
import sys, os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, commalist, pd_load_dataframe, pd_save_dataframe, log


def plt_figure_setup(label = None):
  fig, ax = None, None
  if sys.hexversion < 0x3080000:
    fig, ax = plt.subplots(figsize=np.multiply(plt.rcParams["figure.figsize"], 2), tight_layout=True)
    #fig.suptitle(label)
  else:
    fig, ax = plt.subplots(figsize=np.multiply(plt.rcParams["figure.figsize"], 2), tight_layout=True, label=label)
  plt.set_cmap('Spectral')
  ax.grid(True, 'major', 'y', linestyle='solid')
  ax.grid(True, 'minor', 'y', linestyle='dashed')
  return ax

def pd_chart_waterfall(df, v_0, v_1, d_0, d_1, b_h, b_c, dfv):
  ax = plt_figure_setup('waterfall')
  cmap = plt.get_cmap()
  d_0_1 = np.subtract(d_0, d_1)
  s_0 = np.cumsum(v_0)
  s_1 = np.cumsum(v_1)
  s_0_1 = np.add(np.cumsum(d_0_1), np.max(s_0))
  s_1_0 = np.subtract(s_0_1, d_0)
  l_x = np.reshape(np.stack((np.arange(0.5,df.shape[0]), np.arange(1.5,df.shape[0]+1))), (-1,), order='F')
  b_x = np.concatenate([np.zeros(df.shape[0]), np.arange(0.75, df.shape[0]), np.arange(1.25, df.shape[0]+1), np.full(df.shape[0], df.shape[0] + 1)])
  b_b = np.concatenate((np.subtract(s_0, v_0), s_1_0, s_0_1, np.subtract(s_1, v_1)))
  # waterfall bars
  b_p = ax.bar(b_x, b_h, 0.5, bottom=b_b, color=cmap(b_c), edgecolor='w')
  if sys.hexversion < 0x3080000:
    # polyfill for lack of bar_label
    for i in range(b_h.size):
      ax.annotate(b_h[i], [b_x[i] + 0.1, b_b[i] + b_h[i] * 0.5])
      
  else:
    ax.bar_label(b_p, label_type='center')
  # candelstick bars
  ax.bar(np.arange(1, df.shape[0] + 1), d_0_1, 0.1, bottom=np.subtract(s_0_1, d_0_1), color=np.where(d_0_1 > 0, 'g', 'r'))
  ax.plot(l_x, np.repeat(s_0_1, 2), color='k', ls='-.')
  ax.set_yticks(s_0, minor=False)
  ax.set_yticks(s_1, minor=True)
  ax.set_xticks(np.arange(df.shape[0] + 2))
  # for combatibility with python 3.5 we call set_xtickslabels separately
  ax.set_xticklabels(dfv[:1] + [' '.join(_) if isinstance(_,tuple) else _ for _ in df.index] + dfv[1:])


def pd_chart_compare(df, v_0, v_1, d_0, d_1, b_h, b_c, dfv):
  ax = plt_figure_setup('compare')
  cmap = plt.get_cmap()

  b_d = np.cumsum(np.fmax(v_0, v_1))
  b_a = np.subtract(b_d, np.fmax(v_0, v_1))
  b_x = [0, 0.75, 1.25, 2]
  b_b = np.concatenate((b_a, np.subtract(b_d, d_0), np.subtract(b_d, d_1), b_a))
  b_p = ax.bar(np.repeat(b_x, df.shape[0]), b_h, 0.5, bottom=b_b, color=cmap(b_c), edgecolor='w')
  if sys.hexversion < 0x3080000:
    # polyfill for lack of bar_label
    for i in range(b_h.size):
      ax.annotate(b_h[i], [b_x[int(i / v_0.size % 4)], b_b[i] + b_h[i] * 0.5])
  else:
    ax.bar_label(b_p, label_type='center')

  ax.set_yticks(b_d, minor=False)
  ax.set_yticks(np.subtract(b_d, np.fmax(d_0, d_1)), minor=True)
  ax.set_xticks(b_x, dfv[:1] + ['◩','◪'] + dfv[1:])
  ax.legend(handles=[mpatches.Patch(color=cmap(_)) for _ in b_c], labels=[' '.join(np.atleast_1d(_)) for _ in df.index])

def pd_ijk_merge_compare(il, condition):
  dfa = None
  dfv = []
  ks = None
  gs = None
  for i in range(len(il)):
    fn,groups,key,value = il[i]
    gl = groups.split(';')
    bn = os.path.splitext(os.path.basename(fn))[0]
    vi = '%s:%s:%d' % (bn, value, i)
    dfv.append(vi)
    if gs is None:
      gs = ['g_%d' % i for i in range(len(gl))]

    kl = []
    if key:
      kl.extend(key.split(';'))
    if ks is None:
      ks = ['k_%d' % j for j in range(len(kl))]
    df = pd_load_dataframe(fn, condition, None, [value] + kl + gl)
    if not value:
      value = vi
      df[value] = 1
    df.rename(columns=dict(((value, vi),) + tuple(zip(kl, ks)) + tuple(zip(gl, gs))), inplace=True)
    df = df.pivot_table(vi, ks + gs, None, 'sum')
    if sys.hexversion < 0x3080000 and df.ndim == 1:
      # pd.merge in older versions only works on dataframes
      df = df.to_frame()
    if dfa is None:
      dfa = df
    else:
      dfa = pd.merge(dfa, df, 'outer', left_index=True, right_index=True)
  dfa.reset_index(inplace=True)
  for i0 in range(len(il)):
    i1 = (i0 + 1) % len(il)
    dfa['d_%d' % i0] = np.fmax(np.subtract(np.nan_to_num(dfa[dfv[i1]]), np.nan_to_num(dfa[dfv[i0]])), 0)
  
  return dfa, dfv, ks, gs


def db_buildup(input_a, keys_a, groups_a, value_a, input_b, keys_b, groups_b, value_b, condition, mode, output_table, output_chart, display):
  il = [[input_a, groups_a, keys_a, value_a],[input_b, groups_b, keys_b, value_b]]

  dfa, dfv, ks, gs = pd_ijk_merge_compare(il, condition)
  
  df = dfa.pivot_table(dfv + ['d_0', 'd_1'], gs, aggfunc='sum')

  v_0 = np.nan_to_num(np.ravel(df[dfv[0]]))
  v_1 = np.nan_to_num(np.ravel(df[dfv[1]]))
  d_0 = np.ravel(df['d_0'])
  d_1 = np.ravel(df['d_1'])
  b_h = np.concatenate((v_0,d_0,d_1,v_1))
  b_c = np.tile(np.linspace(0, 1, df.shape[0]), 4)
  
  # multi page pdf
  pdf = None
  if len(output_chart):
    pdf = PdfPages(output_chart)

  if mode in ('waterfall','all'):
    pd_chart_waterfall(df, v_0, v_1, d_0, d_1, b_h, b_c, dfv)
    if pdf is not None:
      pdf.savefig()

  if mode in ('compare','all'):
    pd_chart_compare(df, v_0, v_1, d_0, d_1, b_h, b_c, dfv)
    if pdf is not None:
      pdf.savefig()
  
  if output_table:
    pd_save_dataframe(dfa, output_table)
  
  if pdf is not None:
    pdf.close()
  
  if int(display) and mode not in ('none',''):
    plt.show()

  log("finished")

main = db_buildup

if __name__=="__main__":
  usage_gui(__doc__)
