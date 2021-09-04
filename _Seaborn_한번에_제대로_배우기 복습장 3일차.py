#!/usr/bin/env python
# coding: utf-8

# # Seaborn 한번에 제대로 배우기
# 

# ## Seaborn 특징
# 
# * 여러 변수 간의 관계를 검사하기 위한 데이터 집합 지향 API
# * 범주형 변수를 사용하여 관측치 또는 집계 통계량을 표시하기 위한 전문적인 지원
# * 일변량 또는 이변량 분포를 시각화하고 데이터의 부분 집합 간 비교하기 위한 옵션
# * 서로 다른 종류의 종속 변수에 대한 선형 회귀 모형의 자동 추정 및 표시
# * 복잡한 데이터셋의 전체 구조에 대한 편리한 보기
# * 복잡한 시각화를 쉽게 구축할 수 있는 다중 플롯 그리드 구조를 위한 높은 수준의 추상화
# * 여러 테마가 내장된 matplotlib 그림 스타일링 제어
# * 데이터의 패턴을 충실히 나타내는 색상 팔레트 선택 도구

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.__version__
sns.set(style='whitegrid')


# ## 산점도(Scatter Plot)

# In[2]:


penguins = sns.load_dataset('penguins')
penguins


# In[3]:


sns.relplot(x='bill_length_mm', y='bill_depth_mm', data=penguins)


# In[4]:


sns.relplot(x='flipper_length_mm', y='body_mass_g', data=penguins)


# In[5]:


sns.relplot(x='bill_length_mm', y='bill_depth_mm',
           hue='bill_length_mm', data=penguins)


# In[6]:


sns.relplot(x='flipper_length_mm', y='body_mass_g',
           hue='species', style='species', data=penguins)


# In[7]:


sns.relplot(x='flipper_length_mm', y='body_mass_g',
           hue='species', style='island', data=penguins)


# In[8]:


sns.relplot(x='flipper_length_mm', y='body_mass_g',
           hue='species', col='island', data=penguins)


# In[9]:


sns.relplot(x='bill_length_mm', y='bill_depth_mm',
           hue='flipper_length_mm', data=penguins)


# In[10]:


sns.relplot(x='bill_length_mm', y='bill_depth_mm',
           hue='body_mass_g', data=penguins)


# In[11]:


sns.relplot(x='bill_length_mm', y='bill_depth_mm',
           hue='body_mass_g', size='body_mass_g', data=penguins)


# In[12]:


sns.relplot(x='bill_length_mm', y='bill_depth_mm',
           hue='body_mass_g', size='body_mass_g',
            sizes=(10, 300), data=penguins)


# ## 라인 플롯(Line Plot)

# In[13]:


flights = sns.load_dataset('flights')
flights


# In[14]:


sns.relplot(x='year', y='passengers', kind='line', data=flights)


# In[15]:


dots = sns.load_dataset('dots')
dots


# In[16]:


sns.relplot(x='time', y='firing_rate',
           kind='line', data=dots)


# In[17]:


sns.relplot(x='time', y='firing_rate',
           kind='line', ci=None, data=dots)


# In[18]:


sns.relplot(x='time', y='firing_rate',
           kind='line', ci='sd', data=dots)


# In[19]:


sns.relplot(x='time', y='firing_rate',
           kind='line', estimator=None, data=dots)


# In[20]:


sns.relplot(x='time', y='firing_rate',
           kind='line', hue='choice', data=dots)


# In[21]:


sns.relplot(x='time', y='firing_rate',
           kind='line', hue='align', data=dots)


# In[22]:


sns.relplot(x='time', y='firing_rate',
           kind='line', hue='align',
            style='choice', data=dots)


# In[23]:


sns.relplot(x='time', y='firing_rate',
           kind='line', hue='align',
            dashes=False, markers=True,
            style='choice', data=dots)


# In[24]:


sns.relplot(x='time', y='firing_rate',
           kind='line', hue='align',
            col='choice', data=dots)


# In[25]:


sns.relplot(x='time', y='firing_rate',
            style='choice', kind='line',
            data=dots.query("align == 'sacc'"))


# In[26]:


sns.relplot(x='time', y='firing_rate',
            style='choice', kind='line',
            hue='coherence', data=dots.query("align == 'sacc'"))


# In[27]:


sns.relplot(x='time', y='firing_rate',
            col='choice', kind='line',
            hue='coherence', data=dots.query("align == 'sacc'"))


# In[28]:


fmri = sns.load_dataset('fmri')
fmri


# In[29]:


sns.relplot(x='timepoint', y='signal',
           kind='line', data=fmri)


# In[30]:


sns.relplot(x='timepoint', y='signal',
            sort=False,
           kind='line', data=fmri)


# In[31]:


sns.relplot(x='timepoint', y='signal',
            style='region', size='event',
           kind='line', data=fmri)


# In[32]:


sns.relplot(x='timepoint', y='signal',
            hue='subject',
            style='region', size='event',
           kind='line', data=fmri)


# In[33]:


sns.relplot(x='timepoint', y='signal',
            hue='subject', col='region',
            style='region', size='event',
           kind='line', data=fmri)


# In[34]:


palette = sns.cubehelix_palette(n_colors=14, light=0.8)
sns.relplot(x='timepoint', y='signal',
            hue='subject', col='region',
            style='event',
           palette=palette, kind='line', data=fmri)


# In[35]:


sns.relplot(x='timepoint', y='signal',
            hue='subject', col='region',
           palette=palette, kind='line', data=fmri.query("event=='cue'"))


# In[36]:


sns.relplot(x='timepoint', y='signal',
            hue='subject', col='region', row='event',
           palette=palette, kind='line', data=fmri)


# In[37]:


sns.relplot(x='timepoint', y='signal',
            hue='event', col='subject',
            style='event', col_wrap=4, linewidth=3,
            kind='line', data=fmri.query("event=='cue'"))


# In[38]:


tdf = pd.DataFrame(np.random.randn(40, 4),
                  index=pd.date_range('2020-01-01', periods=40),
                  columns=['A', 'B', 'C', 'D'])
tdf


# In[39]:


sns.relplot(kind='line', data=tdf)


# In[40]:


g = sns.relplot(kind='line', data=tdf)
g.fig.autofmt_xdate()


# In[41]:


g = sns.relplot(kind='line', data=tdf['A'])
g.fig.autofmt_xdate()


# ## 범주형 데이터(Categorical Data)

# ### 범주형 산점도(Categorical scatterplots)
# 
# * `stripplot()` (with `kind="strip"`; the default)
# * `swarmplot()` (with `kind="swarm"`)

# In[42]:


penguins


# In[43]:


sns.catplot(x='species', y='body_mass_g', data=penguins)


# In[44]:


sns.catplot(x='species', y='body_mass_g', 
            jitter=False, data=penguins)


# In[45]:


sns.catplot(x='species', y='body_mass_g', 
            kind='swarm', data=penguins)


# In[46]:


sns.catplot(x='species', y='body_mass_g', 
            hue='sex',
            kind='swarm', data=penguins)


# In[47]:


sns.catplot(x='sex', y='body_mass_g', 
            hue='species',
            kind='swarm', data=penguins)


# In[48]:


sns.catplot(x='sex', y='body_mass_g', 
            hue='species', kind='swarm',
            order=['Female', 'Male'], data=penguins)


# In[49]:


sns.catplot(x='body_mass_g', y='species', 
            hue='island',
            kind='swarm', data=penguins)


# In[50]:


sns.catplot(x='species', y='body_mass_g', 
            hue='sex', col='island', aspect=0.7,
            kind='swarm', data=penguins)


# ### 범주형 분포도(Categorical distribution plots):
# 
# * `boxplot()` (with `kind="box"`)
# * `boxenplot()` (with `kind="boxen"`)
# * `violinplot()` (with `kind="violin"`)

# #### 박스 플롯(Box plots)

# In[51]:


sns.catplot(x='species', y='body_mass_g',
          kind='box', data=penguins)


# In[52]:


sns.catplot(x='species', y='body_mass_g',
          hue='sex',kind='box', data=penguins)


# In[53]:


sns.catplot(x='species', y='body_mass_g',
            hue='sex', kind='box',
            dodge=False, data=penguins)


# In[54]:


sns.catplot(x='species', y='body_mass_g',
            col='sex', kind='box',
            data=penguins)


# In[55]:


iris = sns.load_dataset('iris')
iris


# In[56]:


sns.catplot(kind='box', data=iris)


# In[57]:


sns.catplot(kind='box', orient='h', data=iris)


# In[58]:


sns.catplot(x='species', y='sepal_length', kind='box', data=iris)


# In[59]:


sns.catplot(x='petal_length', y='species', kind='box', data=iris)


# #### 박슨 플롯(Boxen plots)

# In[60]:


diamonds = sns.load_dataset('diamonds')
diamonds


# In[61]:


sns.catplot(x='cut', y='price',
           kind='boxen', data=diamonds)


# In[62]:


sns.catplot(x='color', y='price',
           kind='boxen', data=diamonds)


# In[63]:


sns.catplot(x='color', y='price',
           kind='boxen', data=diamonds.sort_values('color'))


# In[64]:


sns.catplot(x='clarity', y='price',
           kind='boxen', data=diamonds)


# #### 바이올린 플롯(Violin plots)
# 
# * `violinplot`: 커널 밀도 추정과 상자 도표 결합

# In[65]:


penguins


# In[66]:


sns.catplot(x='species', y='body_mass_g',
           hue='sex', kind='violin', data=penguins)


# In[67]:


sns.catplot(x='species', y='body_mass_g',
           hue='sex', kind='violin', 
            bw=.15, cut=0, data=penguins)


# In[68]:


sns.catplot(x='species', y='body_mass_g',
           hue='sex', kind='violin', 
            split=True, data=penguins)


# In[69]:


sns.catplot(x='species', y='body_mass_g',
           hue='sex', kind='violin', 
            inner='stick', split=True, data=penguins)


# In[70]:


g = sns.catplot(x='species', y='body_mass_g',
            kind='violin', 
            inner=None, data=penguins)
sns.swarmplot(x='species', y='body_mass_g',
             color='k', size=3,
             data=penguins, ax=g.ax)


# In[71]:


sns.catplot(kind='violin', data=iris)


# In[72]:


sns.catplot(kind='violin', orient='h', data=iris)


# In[73]:


sns.catplot(x='species', y='sepal_length',
            kind='violin', data=iris)


# In[74]:


sns.catplot(x='petal_length', y='species',
            kind='violin', data=iris)


# ### 범주형 추정치 도표(Categorical estimate plots)
# 
# * `barplot()` (with `kind="bar"`)
# * `pointplot()` (with `kind="point"`)
# * `countplot()` (with `kind="count"`)

# #### 막대 플롯(Bar plots)

# In[75]:


mpg = sns.load_dataset('mpg')
mpg


# In[76]:


sns.catplot(x='origin', y='mpg',
           hue='cylinders', kind='bar',
           data=mpg)


# In[77]:


sns.catplot(x='origin', y='mpg',
           hue='cylinders', kind='bar',
           palette='ch:.20', data=mpg)


# In[78]:


sns.catplot(x='cylinders', y='horsepower',
            kind='bar', palette='ch:.20', edgecolor='.6', 
            data=mpg)


# #### 포인트 플롯(Point plots)
# 
# * 축의 높이를 사용하여 추정값을 인코딩하여 점 추정값과 신뢰 구간 표시

# In[79]:


titanic = sns.load_dataset('titanic')
titanic


# In[80]:


sns.catplot(x='who', y='survived',
           hue='class', kind='point',
           data=titanic)


# In[81]:


sns.catplot(x='class', y='survived',
           hue='who', kind='point',
           data=titanic)


# In[82]:


sns.catplot(x='class', y='survived', hue='who',
            palette={"man":"b", "woman":"r", "child":"g"},
            markers=["^", "o", "."], linestyles=["-", "--", ":"],
            kind='point',data=titanic)


# In[83]:


sns.catplot(x='embark_town', y='survived',
           hue='who', kind='point',
           data=titanic)


# #### 카운트 플롯(Count plots)

# In[84]:


sns.countplot(y='deck', data=titanic)


# In[85]:


sns.countplot(y='embark_town', data=titanic)


# In[86]:


sns.countplot(y='class', data=titanic)


# ## 분포 시각화(Distribution Visualization)

# ### 일변량 분포(Univariate distributions)

# In[87]:


x = np.random.randn(200)
sns.distplot(x);


# #### 히스토그램(Histograms)

# In[88]:


sns.distplot(x, kde=False, rug=True);


# In[89]:


sns.distplot(x, bins=20, kde=False, rug=True);


# #### 커널 밀도 추정(Kernel density estimation)

# In[90]:


sns.distplot(x, hist=False, rug=True);


# In[91]:


sns.kdeplot(x, shade=True);


# In[92]:


sns.kdeplot(x)
sns.kdeplot(x, bw=.2, label='bw:0.2')
sns.kdeplot(x, bw=.1, label='bw: 1')
plt.legend();


# In[93]:


sns.kdeplot(x, shade=True, cut=0)
sns.rugplot(x)


# In[94]:


x = np.random.gamma(10, size=500)
sns.distplot(x, kde=False, fit=stats.gamma)


# ### 이변량 분포(Bivariate distributions)

# #### 산점도(Scatterplots)
# 
# * `jointplot`: 두 개의 변수 간의 이변량(또는 joint) 관계와 별도의 축에 각각의 일변량(또는 marginal) 분포가 모두 표시되는 다중 패널 플롯 생성

# In[95]:


mean = [0, 1]
cov = [(1, .3), (.3, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=['x', 'y'])

sns.jointplot(x='x', y='y', data=df)


# #### 육각 빈 플롯(Hexbin plots)

# In[96]:


x,y = np.random.multivariate_normal(mean, cov, 2000).T
with sns.axes_style('white'):
    sns.jointplot(x=x, y=y, kind='hex');


# #### 커널 밀도 추정(Kernel density estimation)

# In[97]:


sns.jointplot(x='x', y='y', data=df, kind='kde')


# In[98]:


sns.kdeplot(df.x, df.y)
sns.rugplot(df.x, color='r')
sns.rugplot(df.y, color='g', vertical=True)


# In[99]:


cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=60, shade=True)


# In[100]:


g = sns.jointplot(x='x', y='y', data=df, kind='kde')
g.plot_joint(plt.scatter, s=20, linewidth=1, marker='.')
g.ax_joint.collections[0].set_alpha(0)


# ### 페어와이즈 관계 시각화(Visualizing pairwise relationships)

# In[101]:


penguins


# In[102]:


sns.pairplot(penguins);


# In[103]:


sns.pairplot(penguins, hue='species')


# In[104]:


g = sns.PairGrid(penguins)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6)


# ### 히트맵(Heat Map) & 클러스터맵(Cluster Map)

# In[105]:


udata = np.random.rand(20, 30)
sns.heatmap(udata)


# In[106]:


sns.heatmap(udata, vmin=0, vmax=1);


# In[107]:


ndata = np.random.randn(20, 30)
sns.heatmap(ndata, center=0);


# In[108]:


flights = flights.pivot('month', 'year', 'passengers')
sns.heatmap(flights)


# In[109]:


sns.heatmap(flights, annot=True, fmt='d')


# In[110]:


sns.heatmap(flights, linewidths = .2);


# In[111]:


sns.heatmap(flights, cmap='BuPu')


# In[112]:


sns.heatmap(flights, cbar=False)


# In[113]:


grid_kws = {'height_ratios' : (.9, 0.01), 'hspace': .5}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
ax = sns.heatmap(flights, ax=ax,
                cbar_ax = cbar_ax,
                cbar_kws={'orientation': 'horizontal'})


# In[114]:


brain_networks = sns.load_dataset('brain_networks', header=[0, 1, 2], index_col=0)
brain_networks


# In[115]:


networks = brain_networks.columns.get_level_values('network')
used_networks = np.arange(1, 18)
used_columns = (networks.astype(int).isin(used_networks))
brain_networks = brain_networks.loc[:, used_columns]

network_pal = sns.husl_palette(17, s=.5)
network_lut = dict(zip(map(str, used_networks), network_pal))
network_colors = pd.Series(networks, index=brain_networks.columns).map(network_lut)

sns.clustermap(brain_networks.corr(), center=0, cmap='RdBu_r',
              row_colors=network_colors, col_colors=network_colors,
              linewidth=.5, figsize=(12,12))


# ## 선형 관계 시각화(Visualizing linear relationships)

# ### 선형 회귀 모델 시각화 함수

# In[116]:


penguins


# In[117]:


sns.regplot(x='flipper_length_mm', y='body_mass_g',
           data=penguins)


# In[118]:


sns.lmplot(x='flipper_length_mm', y='body_mass_g',
           data=penguins)


# In[119]:


sns.lmplot(x='bill_length_mm', y='body_mass_g',
           hue='island', data=penguins)


# In[120]:


sns.lmplot(x='bill_length_mm', y='body_mass_g',
           col='sex', hue='island', data=penguins)


# In[121]:


sns.lmplot(x='bill_length_mm', y='flipper_length_mm',
            data=penguins)


# In[122]:


sns.lmplot(x='bill_length_mm', y='flipper_length_mm',
           hue='species', data=penguins)


# In[123]:


sns.lmplot(x='bill_length_mm', y='flipper_length_mm',
           hue='species', x_estimator=np.mean, data=penguins)


# In[124]:


sns.lmplot(x='bill_length_mm', y='flipper_length_mm',
           col='island', row='sex', hue='species', data=penguins)


# ### 다른 종류의 모델

# In[125]:


anscombe = sns.load_dataset('anscombe')
anscombe.describe()


# In[126]:


sns.lmplot(x='x', y='y', data=anscombe.query("dataset=='I'")
          , ci=None, scatter_kws={'s':80});


# In[127]:


sns.lmplot(x='x', y='y', data=anscombe.query("dataset=='II'")
          , ci=None, scatter_kws={'s':80});


# In[128]:


sns.lmplot(x='x', y='y', data=anscombe.query("dataset=='II'")
          , order=2,  ci=None, scatter_kws={'s':80});


# In[129]:


sns.lmplot(x='x', y='y', data=anscombe.query("dataset=='III'")
           ,ci=None, scatter_kws={'s':80});


# In[130]:


sns.lmplot(x='x', y='y', data=anscombe.query("dataset=='III'")
           ,robust=True, ci=None, scatter_kws={'s':80});


# In[131]:


penguins


# In[132]:


penguins['long_bill'] = (penguins.bill_length_mm > penguins['bill_length_mm'].mean())


# In[133]:


sns.lmplot(x='body_mass_g', y='long_bill',
          y_jitter=.03, data=penguins)


# In[134]:


sns.lmplot(x='body_mass_g', y='long_bill',
          logistic=True, y_jitter=.03, data=penguins)


# In[135]:


sns.lmplot(x='body_mass_g', y='flipper_length_mm',
          lowess=True, data=penguins)


# In[136]:


sns.residplot(x='x', y='y', data=anscombe.query('dataset == "I"'),
             scatter_kws={'s': 80});


# In[137]:


sns.residplot(x='x', y='y', data=anscombe.query('dataset == "II"'),
             scatter_kws={'s': 80});


# ### 다른 상황의 회귀

# In[138]:


penguins


# In[139]:


sns.jointplot(x='body_mass_g', y='flipper_length_mm',
            kind='reg', data=penguins)


# In[140]:


sns.pairplot(penguins,
            x_vars=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm'],
            y_vars=['body_mass_g'],
             height=4, aspect=.8,
             kind='reg');


# In[141]:


sns.pairplot(penguins,
            x_vars=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm'],
            y_vars=['body_mass_g'], hue='species',
             height=4, aspect=.8,
             kind='reg');


# ## 구조화된 다중 플롯 그리드

# ### FacetGrid

# In[142]:


penguins


# In[143]:


sns.set(style='ticks')


# In[144]:


g=sns.FacetGrid(penguins, col='sex')


# In[145]:


g=sns.FacetGrid(penguins, col='sex')
g.map(plt.hist, 'body_mass_g')


# In[146]:


g=sns.FacetGrid(penguins, col='species')
g.map(plt.hist, 'body_mass_g')


# In[147]:


g=sns.FacetGrid(penguins, col='species', hue='sex')
g.map(plt.hist, 'body_mass_g')


# In[148]:


g=sns.FacetGrid(penguins, col='species', hue='sex')
g.map(plt.scatter, 'bill_length_mm', 'bill_depth_mm', alpha=0.7)
g.add_legend()


# In[149]:


g=sns.FacetGrid(penguins, col='species', hue='sex', margin_titles=True)
g.map(sns.regplot, 'bill_length_mm', 'bill_depth_mm')
g.add_legend()


# In[150]:


g=sns.FacetGrid(penguins, col='species', height=4, aspect=.5)
g.map(sns.barplot, 'sex', 'body_mass_g', order=['Female', 'Male'])


# In[151]:


tips = sns.load_dataset('tips')
tips


# In[152]:


ordered_times = tips.time.value_counts().index
g = sns.FacetGrid(tips, row='time', row_order=ordered_times,
                 height=2, aspect=2)
g.map(sns.distplot, 'tip', hist=False, rug=True)


# In[153]:


g = sns.FacetGrid(tips, hue='day', height=5)
g.map(plt.scatter, 'total_bill', 'tip', s=30, alpha=.7, linewidth=.5)
g.add_legend()


# In[154]:


g = sns.FacetGrid(tips, hue='sex', palette='BuPu',
                  height=5, hue_kws={'marker': ['^', 'v']})
g.map(plt.scatter, 'total_bill', 'tip', s=30, alpha=.7, linewidth=.5)
g.add_legend()


# In[155]:


g = sns.FacetGrid(tips, col='day', col_wrap=2, height=4)
g.map(sns.pointplot, 'sex', 'tip', order=['Female', 'Male'])


# In[156]:


with sns.axes_style('darkgrid'):
    g = sns.FacetGrid(tips, row='sex', col='day', margin_titles=True, height=2.5)
g.map(plt.scatter, 'total_bill', 'tip')


# In[157]:


g = sns.FacetGrid(tips, col='time', margin_titles=True, height=4)
g.map(plt.scatter, 'total_bill', 'tip')
for ax in g.axes.flat:
    ax.plot((0, 50), (0, .2 * 50), c='.2', ls=':')


# In[158]:


r = np.linspace(0, 10, num=100)
df = pd.DataFrame({'r' : r, 'slow' : r, 'medium' : 2 * r, 'fast' : 4 * r})
df = pd.melt(df, id_vars=['r'], var_name='speed', value_name='theta')

g = sns.FacetGrid(df, col='speed', hue='speed',
                 subplot_kws=dict(projection='polar'), height=5,
                 sharex=False, sharey=False, despine=False)
g.map(sns.scatterplot, 'theta', 'r')


# ### 커스텀 함수(Custom functions)

# In[159]:


def quantile_plot(x, **kwargs):
    qntls, xr = stats.probplot(x, fit=False)
    plt.scatter(xr, qntls, **kwargs)
    
g = sns.FacetGrid(tips, col='time', height=4)
g.map(quantile_plot, 'total_bill')


# In[160]:


def qqplot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)
    
g = sns.FacetGrid(tips, col='sex', height=4)
g.map(qqplot, 'total_bill', 'tip')


# In[161]:


g = sns.FacetGrid(tips, col='sex', hue='day', height=4)
g.map(qqplot, 'total_bill', 'tip', s=30, edgecolor='w')
g.add_legend()


# In[162]:


g = sns.FacetGrid(tips, col='time', hue='sex', height=4,
                 hue_kws={'marker':['^', 'o']})
g.map(qqplot, 'total_bill', 'tip', s=30, edgecolor='w')
g.add_legend()


# In[163]:


def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=20, cmap=cmap, **kwargs)
    
with sns.axes_style('dark'):
    g = sns.FacetGrid(tips, hue='sex', col='sex', height=4)
g.map(hexbin, 'total_bill', 'tip', extent=[0, 50, 0, 10])


# ### 페어와이즈 데이터 관계(pairwise data relationships)

# In[164]:


g = sns.PairGrid(tips)
g.map(plt.scatter);


# In[165]:


g = sns.PairGrid(tips)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);


# In[166]:


g = sns.PairGrid(tips, hue='day')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();


# In[167]:


g = sns.PairGrid(tips, vars=['total_bill', 'tip'], hue='day')
g.map(plt.scatter);


# In[168]:


g = sns.PairGrid(tips)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3, legend=False);


# In[169]:


g = sns.PairGrid(tips, y_vars=['tip'],
                x_vars=['size', 'total_bill'], height=4)
g.map(sns.regplot);


# In[170]:


g = sns.PairGrid(tips, hue='day', palette='Set1')
g.map(plt.scatter, s=30, edgecolor='white')
g.add_legend();


# In[171]:


sns.pairplot(tips, hue='day', height=3);


# In[172]:


sns.pairplot(tips, hue='day', palette='Set1', diag_kind='hist',
            height=3);


# ## 그림 미학 제어

# In[173]:


def randplot(flip=1):
    for i in range(1, 7):
        plt.plot(np.random.randn(50).cumsum());


# In[174]:


randplot();


# In[175]:


sns.set()
randplot();


# ### Seaborn 스타일

# In[176]:


sns.set_style('dark')
randplot();


# In[178]:


sns.set_style('white')
randplot();


# In[179]:


sns.set_style('ticks')
randplot();


# In[180]:


sns.set_style('whitegrid')
randplot();


# ### 축 스핀 제거

# In[181]:


d = np.random.randn(50).reshape(10, 5)
d


# In[182]:


sns.distplot(d)
sns.despine();


# In[184]:


sns.violinplot(data=d)
sns.despine(offset=10, trim=True);


# In[185]:


sns.boxplot(data=d, palette='deep')
sns.despine(left=True)


# ### 스타일 임시 설정

# In[186]:


f = plt.figure(figsize=(6, 6))
gs = f.add_gridspec(2, 2)

with sns.axes_style('darkgrid'):
    ax = f.add_subplot(gs[0, 0])
    randplot()
    
with sns.axes_style('white'):
    ax = f.add_subplot(gs[0, 1])
    randplot()
    
with sns.axes_style('ticks'):
    ax = f.add_subplot(gs[1, 0])
    randplot()
    
with sns.axes_style('whitegrid'):
    ax = f.add_subplot(gs[1, 1])
    randplot()
    
f.tight_layout()


# ### 스타일 요소 재정의

# In[187]:


sns.axes_style()


# In[188]:


{'axes.facecolor': 'white',
 'axes.edgecolor': '.8',
 'axes.grid': True,
 'axes.axisbelow': True,
 'axes.labelcolor': '.15',
 'figure.facecolor': 'white',
 'grid.color': '.8',
 'grid.linestyle': '-',
 'text.color': '.15',
 'xtick.color': '.15',
 'ytick.color': '.15',
 'xtick.direction': 'out',
 'ytick.direction': 'out',
 'lines.solid_capstyle': 'round',
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'image.cmap': 'rocket',
 'font.family': ['sans-serif'],
 'font.sans-serif': ['Arial',
  'DejaVu Sans',
  'Liberation Sans',
  'Bitstream Vera Sans',
  'sans-serif'],
 'xtick.bottom': False,
 'xtick.top': False,
 'ytick.left': False,
 'ytick.right': False,
 'axes.spines.left': True,
 'axes.spines.bottom': True,
 'axes.spines.right': True,
 'axes.spines.top': True}


# In[189]:


sns.set_style('darkgrid', {'axes.facecolor':'.5', 'grid.linestyle':':'})
randplot();


# ### 스케일링 플롯 요소

# In[190]:


sns.set()


# In[191]:


sns.set_context('paper')
randplot();


# In[192]:


sns.set_context('talk')
randplot();


# In[193]:


sns.set_context('poster')
randplot();


# In[194]:


sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth':2.5})
randplot();


# ## 컬러 팔레트 선택

# In[195]:


sns.set()


# ### 질적 색상 팔레트

# In[196]:


current_palette = sns.color_palette()
sns.palplot(current_palette)


# #### 원형 컬러 시스템 사용

# In[200]:


sns.palplot(sns.color_palette('hls', 8))
sns.palplot(sns.hls_palette(8, l=.3, s=.8))
sns.palplot(sns.color_palette('husl', 8))


# #### 범주형 컬러 브루어 팔레트 사용

# In[201]:


sns.palplot(sns.color_palette('Paired'))
sns.palplot(sns.color_palette('Set1'))
sns.palplot(sns.color_palette('Set3'))
sns.palplot(sns.color_palette('Set3'))
flatui = ["#99FF00", "#99CC00", "#999900", "#996600", "#993300", "#990000"]
sns.palplot(sns.color_palette(flatui))


# #### xkcd 색상 측량에서 정의된 색상 사용
# 
# * xkcd 색상표: https://xkcd.com/color/rgb/

# In[202]:


plt.plot(np.random.randn(50).cumsum(), sns.xkcd_rgb['pale red'], lw=3)
plt.plot(np.random.randn(50).cumsum(), sns.xkcd_rgb['medium green'], lw=3)
plt.plot(np.random.randn(50).cumsum(), sns.xkcd_rgb['denim blue'], lw=3)


# In[205]:


colors = ['windows blue', 'amber', 'greyish', 'faded green', 'dusty purple']
sns.palplot(sns.xkcd_palette(colors))


# ### 순차 색상 팔레트

# In[206]:


sns.palplot(sns.color_palette('Blues'))
sns.palplot(sns.color_palette('BuGn_r'))
sns.palplot(sns.color_palette('GnBu_d'))


# #### 순차적 입방체 팔레트

# In[207]:


sns.palplot(sns.color_palette('cubehelix', 8))
sns.palplot(sns.cubehelix_palette(8))
sns.palplot(sns.cubehelix_palette(8, start=.5, rot=-.75))
sns.palplot(sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True))


# In[208]:


x, y = np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=500).T
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.kdeplot(x, y, cmap=cmap, shade=True)


# #### 사용자 정의 순차적 팔레트

# In[209]:


sns.palplot(sns.light_palette('green'))
sns.palplot(sns.light_palette('purple'))
sns.palplot(sns.light_palette('navy', reverse=True))
sns.palplot(sns.light_palette((210, 90, 60), input='husl'))
sns.palplot(sns.light_palette('muted purple', input='xkcd'))


# In[211]:


pal = sns.dark_palette('palegreen', as_cmap=True)
sns.kdeplot(x, y, cmap=pal);


# ### 색상 팔레트 나누기

# In[212]:


sns.palplot(sns.color_palette('BrBG', 7))
sns.palplot(sns.color_palette('RdBu_r', 7))
sns.palplot(sns.color_palette('coolwarm', 7))


# #### 커스텀 분기 팔레트

# In[213]:


sns.palplot(sns.diverging_palette(220, 20, n=7))
sns.palplot(sns.diverging_palette(145, 280, s=85, l=25, n=7))
sns.palplot(sns.diverging_palette(10, 220, sep=80, n=7))
sns.palplot(sns.diverging_palette(255, 133, l=60, n=7, center='dark'))


# ### 기본 색상 표 설정

# In[215]:


sns.set_palette('Set1')
randplot();


# In[216]:


sns.set_palette('Set2')
randplot();

