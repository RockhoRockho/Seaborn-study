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


# In[13]:


sns.relplot(x='bill_length_mm', y='bill_depth_mm',
           hue='flipper_length_mm', data=penguins)


# In[14]:


sns.relplot(x='bill_length_mm', y='bill_depth_mm',
           hue='body_mass_g', data=penguins)


# In[15]:


sns.relplot(x='bill_length_mm', y='bill_depth_mm',
           hue='body_mass_g', size='body_mass_g', data=penguins)


# In[16]:


sns.relplot(x='bill_length_mm', y='bill_depth_mm',
           hue='body_mass_g', size='body_mass_g',
            sizes=(10, 300), data=penguins)


# ## 라인 플롯(Line Plot)

# In[17]:


flights = sns.load_dataset('flights')
flights


# In[18]:


sns.relplot(x='year', y='passengers', kind='line', data=flights)


# In[20]:


dots = sns.load_dataset('dots')
dots


# In[21]:


sns.relplot(x='time', y='firing_rate',
           kind='line', data=dots)


# In[23]:


sns.relplot(x='time', y='firing_rate',
           kind='line', ci=None, data=dots)


# In[24]:


sns.relplot(x='time', y='firing_rate',
           kind='line', ci='sd', data=dots)


# In[25]:


sns.relplot(x='time', y='firing_rate',
           kind='line', estimator=None, data=dots)


# In[27]:


sns.relplot(x='time', y='firing_rate',
           kind='line', hue='choice', data=dots)


# In[28]:


sns.relplot(x='time', y='firing_rate',
           kind='line', hue='align', data=dots)


# In[30]:


sns.relplot(x='time', y='firing_rate',
           kind='line', hue='align',
            style='choice', data=dots)


# In[31]:


sns.relplot(x='time', y='firing_rate',
           kind='line', hue='align',
            dashes=False, markers=True,
            style='choice', data=dots)


# In[32]:


sns.relplot(x='time', y='firing_rate',
           kind='line', hue='align',
            col='choice', data=dots)


# In[34]:


sns.relplot(x='time', y='firing_rate',
            style='choice', kind='line',
            data=dots.query("align == 'sacc'"))


# In[35]:


sns.relplot(x='time', y='firing_rate',
            style='choice', kind='line',
            hue='coherence', data=dots.query("align == 'sacc'"))


# In[36]:


sns.relplot(x='time', y='firing_rate',
            col='choice', kind='line',
            hue='coherence', data=dots.query("align == 'sacc'"))


# In[37]:


fmri = sns.load_dataset('fmri')
fmri


# In[38]:


sns.relplot(x='timepoint', y='signal',
           kind='line', data=fmri)


# In[39]:


sns.relplot(x='timepoint', y='signal',
            sort=False,
           kind='line', data=fmri)


# In[42]:


sns.relplot(x='timepoint', y='signal',
            style='region', size='event',
           kind='line', data=fmri)


# In[43]:


sns.relplot(x='timepoint', y='signal',
            hue='subject',
            style='region', size='event',
           kind='line', data=fmri)


# In[45]:


sns.relplot(x='timepoint', y='signal',
            hue='subject', col='region',
            style='region', size='event',
           kind='line', data=fmri)


# In[48]:


palette = sns.cubehelix_palette(n_colors=14, light=0.8)
sns.relplot(x='timepoint', y='signal',
            hue='subject', col='region',
            style='event',
           palette=palette, kind='line', data=fmri)


# In[50]:


sns.relplot(x='timepoint', y='signal',
            hue='subject', col='region',
           palette=palette, kind='line', data=fmri.query("event=='cue'"))


# In[52]:


sns.relplot(x='timepoint', y='signal',
            hue='subject', col='region', row='event',
           palette=palette, kind='line', data=fmri)


# In[54]:


sns.relplot(x='timepoint', y='signal',
            hue='event', col='subject',
            style='event', col_wrap=4, linewidth=3,
            kind='line', data=fmri.query("event=='cue'"))


# In[56]:


tdf = pd.DataFrame(np.random.randn(40, 4),
                  index=pd.date_range('2020-01-01', periods=40),
                  columns=['A', 'B', 'C', 'D'])
tdf


# In[57]:


sns.relplot(kind='line', data=tdf)


# In[58]:


g = sns.relplot(kind='line', data=tdf)
g.fig.autofmt_xdate()


# In[59]:


g = sns.relplot(kind='line', data=tdf['A'])
g.fig.autofmt_xdate()


# ## 범주형 데이터(Categorical Data)

# ### 범주형 산점도(Categorical scatterplots)
# 
# * `stripplot()` (with `kind="strip"`; the default)
# * `swarmplot()` (with `kind="swarm"`)

# In[60]:


penguins


# In[61]:


sns.catplot(x='species', y='body_mass_g', data=penguins)


# In[62]:


sns.catplot(x='species', y='body_mass_g', 
            jitter=False, data=penguins)


# In[63]:


sns.catplot(x='species', y='body_mass_g', 
            kind='swarm', data=penguins)


# In[64]:


sns.catplot(x='species', y='body_mass_g', 
            hue='sex',
            kind='swarm', data=penguins)


# In[66]:


sns.catplot(x='sex', y='body_mass_g', 
            hue='species',
            kind='swarm', data=penguins)


# In[67]:


sns.catplot(x='sex', y='body_mass_g', 
            hue='species', kind='swarm',
            order=['Female', 'Male'], data=penguins)


# In[68]:


sns.catplot(x='body_mass_g', y='species', 
            hue='island',
            kind='swarm', data=penguins)


# In[69]:


sns.catplot(x='species', y='body_mass_g', 
            hue='sex', col='island', aspect=0.7,
            kind='swarm', data=penguins)


# ### 범주형 분포도(Categorical distribution plots):
# 
# * `boxplot()` (with `kind="box"`)
# * `boxenplot()` (with `kind="boxen"`)
# * `violinplot()` (with `kind="violin"`)

# #### 박스 플롯(Box plots)

# In[71]:


sns.catplot(x='species', y='body_mass_g',
          kind='box', data=penguins)


# In[72]:


sns.catplot(x='species', y='body_mass_g',
          hue='sex',kind='box', data=penguins)


# In[73]:


sns.catplot(x='species', y='body_mass_g',
            hue='sex', kind='box',
            dodge=False, data=penguins)


# In[74]:


sns.catplot(x='species', y='body_mass_g',
            col='sex', kind='box',
            data=penguins)


# In[76]:


iris = sns.load_dataset('iris')
iris


# In[77]:


sns.catplot(kind='box', data=iris)


# In[78]:


sns.catplot(kind='box', orient='h', data=iris)


# In[79]:


sns.catplot(x='species', y='sepal_length', kind='box', data=iris)


# In[80]:


sns.catplot(x='petal_length', y='species', kind='box', data=iris)


# #### 박슨 플롯(Boxen plots)

# In[81]:


diamonds = sns.load_dataset('diamonds')
diamonds


# In[83]:


sns.catplot(x='cut', y='price',
           kind='boxen', data=diamonds)


# In[84]:


sns.catplot(x='color', y='price',
           kind='boxen', data=diamonds)


# In[86]:


sns.catplot(x='color', y='price',
           kind='boxen', data=diamonds.sort_values('color'))


# In[87]:


sns.catplot(x='clarity', y='price',
           kind='boxen', data=diamonds)


# #### 바이올린 플롯(Violin plots)
# 
# * `violinplot`: 커널 밀도 추정과 상자 도표 결합

# In[88]:


penguins


# In[91]:


sns.catplot(x='species', y='body_mass_g',
           hue='sex', kind='violin', data=penguins)


# In[92]:


sns.catplot(x='species', y='body_mass_g',
           hue='sex', kind='violin', 
            bw=.15, cut=0, data=penguins)


# In[93]:


sns.catplot(x='species', y='body_mass_g',
           hue='sex', kind='violin', 
            split=True, data=penguins)


# In[94]:


sns.catplot(x='species', y='body_mass_g',
           hue='sex', kind='violin', 
            inner='stick', split=True, data=penguins)


# In[98]:


g = sns.catplot(x='species', y='body_mass_g',
            kind='violin', 
            inner=None, data=penguins)
sns.swarmplot(x='species', y='body_mass_g',
             color='k', size=3,
             data=penguins, ax=g.ax)


# In[99]:


sns.catplot(kind='violin', data=iris)


# In[100]:


sns.catplot(kind='violin', orient='h', data=iris)


# In[101]:


sns.catplot(x='species', y='sepal_length',
            kind='violin', data=iris)


# In[102]:


sns.catplot(x='petal_length', y='species',
            kind='violin', data=iris)


# ### 범주형 추정치 도표(Categorical estimate plots)
# 
# * `barplot()` (with `kind="bar"`)
# * `pointplot()` (with `kind="point"`)
# * `countplot()` (with `kind="count"`)

# #### 막대 플롯(Bar plots)

# In[104]:


mpg = sns.load_dataset('mpg')
mpg


# In[105]:


sns.catplot(x='origin', y='mpg',
           hue='cylinders', kind='bar',
           data=mpg)


# In[107]:


sns.catplot(x='origin', y='mpg',
           hue='cylinders', kind='bar',
           palette='ch:.20', data=mpg)


# In[109]:


sns.catplot(x='cylinders', y='horsepower',
            kind='bar', palette='ch:.20', edgecolor='.6', 
            data=mpg)


# #### 포인트 플롯(Point plots)
# 
# * 축의 높이를 사용하여 추정값을 인코딩하여 점 추정값과 신뢰 구간 표시

# In[111]:


titanic = sns.load_dataset('titanic')
titanic


# In[112]:


sns.catplot(x='who', y='survived',
           hue='class', kind='point',
           data=titanic)


# In[118]:


sns.catplot(x='class', y='survived',
           hue='who', kind='point',
           data=titanic)


# In[116]:


sns.catplot(x='class', y='survived', hue='who',
            palette={"man":"b", "woman":"r", "child":"g"},
            markers=["^", "o", "."], linestyles=["-", "--", ":"],
            kind='point',data=titanic)


# In[117]:


sns.catplot(x='embark_town', y='survived',
           hue='who', kind='point',
           data=titanic)


# #### 카운트 플롯(Count plots)

# In[120]:


sns.countplot(y='deck', data=titanic)


# In[121]:


sns.countplot(y='embark_town', data=titanic)


# In[122]:


sns.countplot(y='class', data=titanic)


# ## 분포 시각화(Distribution Visualization)

# ### 일변량 분포(Univariate distributions)

# In[123]:


x = np.random.randn(200)
sns.distplot(x);


# #### 히스토그램(Histograms)

# In[125]:


sns.distplot(x, kde=False, rug=True);


# In[127]:


sns.distplot(x, bins=20, kde=False, rug=True);


# #### 커널 밀도 추정(Kernel density estimation)

# In[128]:


sns.distplot(x, hist=False, rug=True);


# In[129]:


sns.kdeplot(x, shade=True);


# In[131]:


sns.kdeplot(x)
sns.kdeplot(x, bw=.2, label='bw:0.2')
sns.kdeplot(x, bw=.1, label='bw: 1')
plt.legend();


# In[132]:


sns.kdeplot(x, shade=True, cut=0)
sns.rugplot(x)


# In[133]:


x = np.random.gamma(10, size=500)
sns.distplot(x, kde=False, fit=stats.gamma)


# ### 이변량 분포(Bivariate distributions)

# #### 산점도(Scatterplots)
# 
# * `jointplot`: 두 개의 변수 간의 이변량(또는 joint) 관계와 별도의 축에 각각의 일변량(또는 marginal) 분포가 모두 표시되는 다중 패널 플롯 생성

# In[ ]:





# #### 육각 빈 플롯(Hexbin plots)

# In[ ]:





# #### 커널 밀도 추정(Kernel density estimation)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 페어와이즈 관계 시각화(Visualizing pairwise relationships)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 히트맵(Heat Map) & 클러스터맵(Cluster Map)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 선형 관계 시각화(Visualizing linear relationships)

# ### 선형 회귀 모델 시각화 함수

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 다른 종류의 모델

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 다른 상황의 회귀

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 구조화된 다중 플롯 그리드

# ### FacetGrid

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 커스텀 함수(Custom functions)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 페어와이즈 데이터 관계(pairwise data relationships)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 그림 미학 제어

# In[ ]:





# In[ ]:





# In[ ]:





# ### Seaborn 스타일

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 축 스핀 제거

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 스타일 임시 설정

# In[ ]:





# ### 스타일 요소 재정의

# In[ ]:





# In[ ]:





# In[ ]:





# ### 스케일링 플롯 요소

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 컬러 팔레트 선택

# In[ ]:





# ### 질적 색상 팔레트

# In[ ]:





# #### 원형 컬러 시스템 사용

# In[ ]:





# #### 범주형 컬러 브루어 팔레트 사용

# In[ ]:





# #### xkcd 색상 측량에서 정의된 색상 사용
# 
# * xkcd 색상표: https://xkcd.com/color/rgb/

# In[ ]:





# In[ ]:





# ### 순차 색상 팔레트

# In[ ]:





# #### 순차적 입방체 팔레트

# In[ ]:





# In[ ]:





# #### 사용자 정의 순차적 팔레트

# In[ ]:





# In[ ]:





# ### 색상 팔레트 나누기

# In[ ]:





# #### 커스텀 분기 팔레트

# In[ ]:





# ### 기본 색상 표 설정

# In[ ]:





# In[ ]:




