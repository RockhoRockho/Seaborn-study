# Seaborn-study
--
## Seaborn study Day 1 (2021-09-02)

- Seaborn 특징 학습
- 산점도(scatter plot) 학습(`relplot()`)
- 라인플롯 학습
    - x, y값 입력
    - `kind`
    - `data` 입력
    - `ci`(편차설정)
    - `sd`(표준편차)
    - `estimator`
    - `hue`
    - `col` 열갯수
    - `style`
    - `dsahes, markers`
    - `query`
    - `sort`
    - `size, sizes`
    - `palette`

- 범주형 산점도 학습(`stripplot()`, `swarmplot()`)
    - `catplot()`
    - catplot의 `kind=swarm`, `kind=box`, `kind=boxen`, `kind=violin`, 'kind=bar`
    - `order`
    - `aspect` 간격
    - `dodge` 겹치기
    - `row` 행갯수
    - `orient` x,y 축 변경
    - violin = `bw`, `cut`, `split`(반으로 분할), `inner`(내부선)
    - `edgecolor`
    - `countplot()`
- 분포 시각화 학습
    - 일변량 분포 학습(`distplot()`)
    - 히스토그램 학습(`kde`, `rug`, `bin`)
    - 커널 밀도 추정 학습(`kdeplot()`, `label`)

--

## Seaborn study Day 2 (2021-09-03)

- 이변량 분포 학습
    - 산점도, 육각 빈 플롯(hexbin plots), 커널 밀도 추정 학습
- 페어와이즈 관계시각화(visualizing pairwise relationships) 학습 (`pairplot()`)
- 히트맵 & 클러스터맵 학습(`heatmap()`, `clustermap()`)
- 선형 관계 시각화
    - 선형 회귀 모델 시각화 함수(`regplot()`, `lmplot()`, `ci`, `order`, `_kws`)
    - `logistic`, `lowess`, `_jitter`
- 구조화된 다중 플롯 그리드(`FacetGrid()`) 학습
- 커스텀 함수 
