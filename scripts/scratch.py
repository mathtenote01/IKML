# %reset -f
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

left   = np.array(['福島', '愛知', '神奈川', '大阪', '東京'])
height = np.array([160, 220, 280, 360, 1820])

# サブプロット2行1列のサブプロットを用意
fig, ax = plt.subplots(nrows=2, figsize=(3,4), dpi=160, sharex='col',
                       gridspec_kw={'height_ratios': (1,2)} )

fig.patch.set_facecolor('white') # 背景色を「白」に設定

ax[0].bar(left,height) # 上段
ax[1].bar(left,height) # 下段

# サブプロット間の上下間隔をゼロに設定
fig.subplots_adjust(hspace=0.0)  

# 下段サブプロット
ax[1].set_ylim(0,400)  # 区間幅 400
ax[1].set_yticks(np.arange(0,300+1,100))

# 上段サブプロット
ax[0].set_ylim(1750,1950)  # 区間幅 200
ax[0].set_yticks((1800,1900))

# 下段のプロット領域上辺を非表示
ax[1].spines['top'].set_visible(False)

# 上段のプロット領域底辺を非表示、X軸の目盛とラベルを非表示
ax[0].spines['bottom'].set_visible(False)
ax[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False) 

## ニョロ線の描画
d1 = 0.02 # X軸のはみだし量
d2 = 0.03 # ニョロ波の高さ
wn = 21   # ニョロ波の数（奇数値を指定）

pp = (0,d2,0,-d2)
px = np.linspace(-d1,1+d1,wn)
py = np.array([1+pp[i%4] for i in range(0,wn)])
p = Path(list(zip(px,py)), [Path.MOVETO]+[Path.CURVE3]*(wn-1))

line1 = mpatches.PathPatch(p, lw=4, edgecolor='black',
                          facecolor='None', clip_on=False,
                          transform=ax[1].transAxes, zorder=10)

line2 = mpatches.PathPatch(p,lw=3, edgecolor='white',
                           facecolor='None', clip_on=False,
                           transform=ax[1].transAxes, zorder=10,
                           capstyle='round')

a = ax[1].add_patch(line1)
a = ax[1].add_patch(line2)