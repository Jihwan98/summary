# About visualization

## Matplotlib
- pyplot 객체를 사용하여 데이터를 표시
- pyplot 객체에 그래프들을 쌓은 다음 show로 flush

  ```python
  import matplotlib.pyplot as plt
  X = range(100)
  Y = [value**2 for value in X]
  plt.plot(X,Y)
  plt.show()
  ```
![image](https://user-images.githubusercontent.com/76936390/111906056-25dc8780-8a92-11eb-8b3a-c1f50d13ebed.png)

- 최대 단점 argument를 kwargs 받음
- 고정된 argument가 없어서 shift + tab 으로 확인이 어려움
- Graph는 원래 figure 객체에 생성됨
- pyplot 객체 사용시, 기본 figure에 그래프가 그려짐

  ```python
  X_1 = range(100)
  Y_1 = [np.cos(value) for value in X]

  X_2 = range(100)
  Y_2 = [np.sin(value) for value in X]

  plt.plot(X_1, Y_1)
  plt.plot(X_2, Y_2)
  plt.plot(range(100),range(100))

  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/76936390/111906486-14947a80-8a94-11eb-8c1c-e72a908de0c7.png)
  ```python
  fig = plt.figure()  # figure 반환
  fig.set_size_inches(10,5)  # 크기 지정
  ax_1 = fig.add_subplot(1,2,1)  # 두개의 plot 생성
  ax_2 = fig.add_subplot(1,2,2)  # 두개의 plot 생성

  ax_1.plot(X_1, Y_1, c="b")  # 첫번째 plot
  ax_2.plot(X_2, Y_2, c="g")  # 두번째 plot

  plt.show()  # show & flush
  ```
  ![image](https://user-images.githubusercontent.com/76936390/111906495-1f4f0f80-8a94-11eb-86b7-c25439834945.png)

  ### Set color
  - color 속성을 사용
  - flat -> 흑백, rgb color, predefined color 사용
  ```python
  plt.plot(X_1, Y_1, color = "#eeefff")
  plt.plot(X_2, Y_2, c = "r")
  ```

  ### Set linestyle
  - ls 또는 linestyle 속성 사용
  ```python
  plt.plot(X_1, Y_1, c="b", linestyle="dashed")
  plt.plot(X_2, Y_2, c="r", ls="dotted")
  ```

  ### Set title
  - pyplot에 title 함수 사용, figure의 subplot별 입력가능
  - Latex 타입의 표현도 가능 (수식 표현 가능)
  - `plt.title("Test")`
  - `plt.title("$y=ax+b$")`

  ### Set legend
  - Legend 함수로 범례를 표시함, loc 위치 등 속성 지정
  - `plt.legend(shadow=False, fancybox=True, loc="lower right")`
  ```python
  plt.plot(X_1, Y_1, c="b", linestyle="dashed", label='line_1')
  plt.plot(X_2, Y_2, c="r", ls="dotted", label='line_1')
  plt.legend(shadow=True, fancybox=True, loc="lower right")

  plt.title("$y=ax+b$")
  plt.xlabel('$x_line$')
  plt.ylabel('y_line')

  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/76936390/111909096-3f380080-8a9f-11eb-93a6-bb4b2cb1b468.png)

  ### Set grid & xylim
  - Graph 보조선을 긋는 grid와 xy축 범위 한계를 지정
  ```python
  plt.grid(True, lw=0.4, ls="--", c=".90")
  plt.xlim(-100, 200)
  plt.ylim(-100, 300)
  ```
  ![image](https://user-images.githubusercontent.com/76936390/111975946-7ca29980-8b44-11eb-9d15-11804b84864c.png)

  ### savefig
  - 이미지 저장
  - `plt.savefig("test.png", c="a")`

  ### Scatter
  - scatter 함수 사용, marker : scatter 모양 지정
  ```python
  data_1 = np.random.rand(512, 2)
  data_2 = np.random.rand(512, 2)

  plt.scatter(data_1[:,0], data_1[:,1], c='b', marker='x')
  plt.scatter(data_2[:,0], data_2[:,1], c='r', marker='o')

  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/76936390/111976642-300b8e00-8b45-11eb-9c2b-072a62affbe1.png)
  
  - s : 데이터의 크기를 지정, 데이터의 크기 비교 가능
  ```python
  N = 50
  x = np.random.rand(N)
  y = np.random.rand(N)
  colors = np.random.rand(N)
  area = np.pi * (15 * np.random.rand(N))**2
  plt.scatter(x, y, s=area, c=colors, alpha=0.5)    # alpha : 투명도
  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/76936390/111976980-95f81580-8b45-11eb-8fcf-d63c038cc68f.png)

  ### Bar chart
  - Bar 함수 사용
  ```python
  data = [[5., 25., 50., 20.],
        [4., 23., 51., 17],
        [6., 22., 52., 19]]

  X = np.arange(0,8,2)

  plt.bar(X + 0.00, data[0], color = 'b', width = 0.50)
  plt.bar(X + 0.50, data[1], color = 'g', width = 0.50)
  plt.bar(X + 1.0, data[2], color = 'r', width = 0.50)
  plt.xticks(X+0.50, ("A","B","C", "D"))
  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/76936390/111977663-57af2600-8b46-11eb-9b8c-215446aba008.png)

  ```python
  data = np.array([[5., 25., 50., 20.],
        [4., 23., 51., 17],
        [6., 22., 52., 19]])

  color_list = ['b', 'g', 'r']
  data_label = ["A","B","C"]
  X = np.arange(data.shape[1])

  for i in range(3):
      plt.bar(X, data[i], bottom = np.sum(data[:i], axis=0), 
              color = color_list[i], label=data_label[i])
  plt.legend()
  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/76936390/111978363-297e1600-8b47-11eb-9de7-8fc9b883e36b.png)
  
  ```python
  A = [5., 30., 45., 22.]
  B = [5, 25, 50, 20]

  X = range(4)

  plt.bar(X, A, color = 'b')
  plt.bar(X, B, color = 'r', bottom = 60)
  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/76936390/111978770-a3160400-8b47-11eb-835b-a7b05fac6d0b.png)
  
  ```python
  women_pop = np.array([5, 30, 45, 22])
  men_pop = np.array([5, 25, 50, 20])
  X = np.arange(4)

  plt.barh(X, women_pop, color = 'r')
  plt.barh(X, -men_pop, color = 'b')
  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/76936390/111978794-add09900-8b47-11eb-97cc-c80020e006f4.png)

  ### histogram
  ```python
  x = np.random.randn(1000)
  plt.hist(x, bins=10)
  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/76936390/111978613-72ce6580-8b47-11eb-8c10-6ed6853bec26.png)

  ### boxplot
  ```python
  data = np.random.randn(100,5)
  plt.boxplot(data)
  plt.show()
  ```
  ![image](https://user-images.githubusercontent.com/76936390/111978700-90033400-8b47-11eb-9f63-5266b88dd0c2.png)
  
  [실행코드보기](./matplotlib.ipynb)


## Seaborn
  - 기존 matplotlib에 기본 설정을 추가
  - 복잡한 그래프를 간단하게 만들 수 있는 wrapper
  - 간단한 코드 + 예쁜 결과
  - matplotlib와 같은 기본적인 plot
  - 손쉬운 설정으로 데이터 산출
  - lineplot, scatterplot, countplot 등
  - `sns.lineplot(x="total_bill", y="tip", data=tips)`
  ```python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  tips = sns.load_dataset("tips")
  fmri = sns.load_dataset("fmri")
  
  sns.set_style("whitegrid")
  sns.lineplot(x="timepoint",y="signal", data=fmri)
  ```
  ![image](https://user-images.githubusercontent.com/76936390/111981331-e0c85c00-8b4a-11eb-8d59-ee66c07bf21f.png)
  
  ### hue
  ```python
  sns.lineplot(x="timepoint",y="signal", hue="event", data=fmri)
  ```
  ![image](https://user-images.githubusercontent.com/76936390/112457607-44e35e00-8d9f-11eb-88f7-a34ec288d0c8.png)
  
  ### scatterplot
  ```python
  sns.scatterplot(x="total_bill", y="tip", data=tips)
  ```
  ![image](https://user-images.githubusercontent.com/76936390/112457774-6c3a2b00-8d9f-11eb-8eca-2619e18f03c9.png)
  
  ```python
  sns.scatterplot(x="total_bill", y="tip", hue="time", data=tips)
  ```
  ![image](https://user-images.githubusercontent.com/76936390/112458080-bfac7900-8d9f-11eb-82fa-a94bab0003ad.png)


  ### regplot
  ```python
  sns.regplot(x="total_bill", y="tip", data=tips)
  ```
  ![image](https://user-images.githubusercontent.com/76936390/112457890-8a079000-8d9f-11eb-81c6-05aea47b83fa.png)
  
  ### countplot
  ```python
  sns.countplot(x="smoker", data=tips)
  ```
  ![image](https://user-images.githubusercontent.com/76936390/112458297-fd110680-8d9f-11eb-9b3e-3356da4c63b9.png)
  
  ```python
  sns.countplot(x="smoker", hue="time", data=tips)
  ```
  ![image](https://user-images.githubusercontent.com/76936390/112458333-0601d800-8da0-11eb-893e-c806d9c060f7.png)

  ### barplot
  ```python
  sns.barplot(x="day", y="total_bill", data=tips)
  ```
  ![image](https://user-images.githubusercontent.com/76936390/112458511-39dcfd80-8da0-11eb-8768-5cb53911a159.png)
  
  ### distplot
  ```python
  sns.distplot(tips["total_bill"], kde=False)
  ```
  ![image](https://user-images.githubusercontent.com/76936390/112458939-a952ed00-8da0-11eb-8a0e-f352d3d0859d.png)
  
 - violinplot
 - swarmpolot
 - catplot
 - pointplot 등등

  [실행코드보기](./seaborn.ipynb)
  
  
## Time Series Data
  ### datetime index
  - python에서 datetime 모듈을 활용
  ```python
  from datetime import datetime
  date_str = '09-19-2018'
  date_object = datetime.strptime(date_str, '%m-%d-%Y').date()
  
  date_str = '2018/09/19'
  date_object = datetime.strptime(date_str, '%Y/%m/%d').date()
  
  date_str = '180919'
  date_object = datetime.strptime(date_str, '%y%m%d').date()
  ```
  ```python
  date_object.day         # 19
  date_object.month       # 9
  date_object.weekday()   # 2
  ```
  - 시,분,초 단위도 역시 가능.
  - datetime 차이도 구할 수 있음. (timedelta type으로 나옴)
  
  ### datetime index 만들기
  - 대부분의 데이터는 str 형태로 되어있음 -> 호출 후 datetime index로 변환이 필요함
  - ```df['datetime'] = pd.to_datetime(df['date'])

  ### Time resampling
  - 시간 기준 데이터로 Aggregation
  - Groupby와 유사 -> 훨씬 간단하고 유용
  ```python
  df["count"].resample('Q').sum()   # 분기별
  df["count"].resample('M').sum()   # 달별
  df["count"].resample('D').sum()   # 일별
  df["count"].resample('W').sum()   # 주별
  ```
  - pd.date_range()
  - time_delta_range()
  - period_range()
  - interval_range()
  - 등등
  ```python
  period = pd.date_range(start='2011-01-01', end='2011-05-31', freq='M')
  df["count"].resample('M').sum()[period]
  
  period = pd.date_range(start='2011-01-01', period=12, freq='M')
  df["count"].resample('M').sum()[period]
  
  df["count"].resample('M').sum()["2011-01-01":"2012-05-01"]
  ```
  - 요일별 데이터 뽑기?
  ```python
  df["dayofweek"] = df.index.dayofweek    # dayofyear, weekofyear 등등
  df.groupby("dayofweek")["count"].mean()
  ```
  ### Time shifting
  - 시간의 차 분석
  - Pandas 내 Time shifting 기능으로 분석
  - `shift`
  ```python
  monthly_avg = df["count"].resample("M").mean()
  monthly_avg.shift(periods=2, fill_value=0)
  ```
  ```python
  monthly_avg = df["count"].resample("M").mean()
  result = []
  for period in range(1,6):
    temp_avg = monthly_avg.shift(periods=period, fill_value=0)
    temp_avg = temp_avg.rename("{}_monthly_shift".format(period))
    result.append(temp_avg)
  
  pd.concat(result, axis=1)
  ```
  
  ### Moving average
  - 시계열 데이터는 노이즈 발생
  - 노이즈를 줄이면서 추세보기, 이동평균법
  - `rolling`
  ```python
  day_avg = df["count"].resample("D").mean()
  day_avg.rolling(window=30).mean().plot()
  ```
  ### Cumsum
  - 시계열 데이터를 window 마다 합침
  - rolling(window=10).sum()과 다름
  ```python
  monthly_avg = df["count"].resample("M").mean()
  cumsum_avg = df["count"].resample("M").mean().cumsum()
  monthly_avg = monthly_avg.rename("monthly_avg")
  cumsum_avg = cumsum_avg.rename("cumsum_avg")
  df_monthly = pd.concat([monthly_avg, cumsum_avg], axis=1)
  ax = df_monthly.plot(y="monthly_avg", use_index = True)
  df_monthly.plot(y="cumsum_avg", secondary_y=True, ax=ax)
  ```
  ![image](https://user-images.githubusercontent.com/76936390/115037763-c3986a80-9f09-11eb-95b6-521602b2a9e0.png)

