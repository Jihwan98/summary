# About Pandas
- 구조화된 데이터의 처리를 지원하는 Python 라이브러리
- panel data -> pandas
- 고성능 array 계산 라이브러리인 numpy와 통합하여, 강력한 "스프레드시트"처리 기능을 제공
- 인덱싱, 연산용 함수, 전처리 함수 등을 제공함
- 데이터 처리 및 통계 분석을 위해 사용


## Series
  - index 가 추가된 numpy이다
  ```python
  from pandas import Series, DataFrame
  import pandas as pd
  
  list_data = [1,2,3,4,5]
  example_obj = Series(data = list_data)
  example_obj
  ```
  ```
  0    1
  1    2
  2    3
  3    4
  4    5
  dtype: int64
  ```
  
  ```python
  list_data = [1,2,3,4,5]
  list_name = ["a","b","c","d","e"]
  example_obj = Series(data = list_data, index = list_name)
  example_obj
  ```
  ```
  a    1
  b    2
  c    3
  d    4
  e    5
  dtype: int64
  ```
  
  ```python
  dic_data = {"a":1 , "b":2, "c":3, "d":4, "e":5}
  example_obj = Series(dic_data, dtype=np.float32, name="example_data")
  example_obj
  ```
  ```
  a    1.0
  b    2.0
  c    3.0
  d    4.0
  e    5.0
  Name: example_data, dtype: float32
  ```
  
  ```python
  example_obj["a"]
  ```
  `1.0`
  ```python
  example_obj["a"] = 3.2
  example_obj  
  ```
  ```
  a    3.2
  b    2.0
  c    3.0
  d    4.0
  e    5.0
  Name: example_data, dtype: float32
  ```
  
  ```python
  example_obj.values
  ```
  `array([3.2, 2. , 3. , 4. , 5. ], dtype=float32)`
  ```python
  example_obj.index
  ```
  `Index(['a', 'b', 'c', 'd', 'e'], dtype='object')`
  ```python
  example_obj.name = "number"
  example_obj.index.name = "alphabet"
  example_obj
  ```
  ```
  alphabet
  a    3.2
  b    2.0
  c    3.0
  d    4.0
  e    5.0
  Name: number, dtype: float32
  ```
  
  [실행코드보기](./series.ipynb)
  
  
## DataFrame
  - DataFrame 생성
  ```python
  raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
        'age': [42, 52, 36, 24, 73],
        'city': ['San Francisco', 'Baltimore', 'Miami', 'Douglas', 'Boston']}
  df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'city'])
  df
  ```
  ```
  first_name	last_name	age	city
  0	Jason	Miller	42	San Francisco
  1	Molly	Jacobson	52	Baltimore
  2	Tina	Ali	36	Miami
  3	Jake	Milner	24	Douglas
  4	Amy	Cooze	73	Boston
  ```
  - 원하는 column 만 추출
  ```python
  DataFrame(raw_data, columns = ["age", "city"])
  ```
  ```
  age	city
  0	42	San Francisco
  1	52	Baltimore
  2	36	Miami
  3	24	Douglas
  4	73	Boston
  ```
  - column 추가
  ```python
  DataFrame(raw_data, columns = ["first_name","last_name","age", "city", "debt"])
  ```
  ```
  	first_name	last_name	age	city	debt
  0	Jason	Miller	42	San Francisco	NaN
  1	Molly	Jacobson	52	Baltimore	NaN
  2	Tina	Ali	36	Miami	NaN
  3	Jake	Milner	24	Douglas	NaN
  4	Amy	Cooze	73	Boston	NaN
  ```
  - column 선택 후 series 추출
  ```python
  df = DataFrame(raw_data, columns = ["first_name", "last_name", "age", "city", "debt"])
  df.first_name    # == df["first_name"]
  ```
  ```
  0    Jason
  1    Molly
  2     Tina
  3     Jake
  4      Amy
  Name: first_name, dtype: object
  ```
  - loc - index location
  - iloc - index position
  ```python
  df.loc[1]
  ```
  ```
  first_name        Molly
  last_name      Jacobson
  age                  52
  city          Baltimore
  debt                NaN
  Name: 1, dtype: object
  ```
  ```python
  df["age"].iloc[1:]
  ```
  ```
  1    52
  2    36
  3    24
  4    73
  Name: age, dtype: int64
  ```
  - loc은 index 이름, iloc은 index number
  ```python
  s = pd.Series(np.nan, index=[49,48,47,46,45, 1, 2, 3, 4, 5])
  s.loc[:3]
  ```
  ```
  49   NaN
  48   NaN
  47   NaN
  46   NaN
  45   NaN
  1    NaN
  2    NaN
  3    NaN
  dtype: float64
  ```
  ```python
  s.iloc[:3]
  ```
  ```
  49   NaN
  48   NaN
  47   NaN
  dtype: float64
  ```
  - column에 새로운 데이터 할당
  ```python
  df.debt = df.age > 40
  df
  ```
  ```
  	first_name	last_name	age	city	debt
  0	Jason	Miller	42	San Francisco	True
  1	Molly	Jacobson	52	Baltimore	True
  2	Tina	Ali	36	Miami	False
  3	Jake	Milner	24	Douglas	False
  4	Amy	Cooze	73	Boston	True
  ```
  - transpose `df.T`
  - 값 출력 `df.values`
  - csv 변환 `df.to_csv()`
  - column 삭제 `del df["debt"]`
  
  ```python
  pop = {'Nevada': {2001: 2.4, 2002: 2.9},
 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}

  DataFrame(pop)
  ```
  ```
  	Nevada	Ohio
  2001	2.4	1.7
  2002	2.9	3.6
  2000	NaN	1.5
  ```
  
  [실행코드보기](./DataFrame.ipynb)

## Selection & Drop
  ### Selection with column names
  ```python
  df = pd.read_excel("./data/excel-comp-data.xlsx")
  df.head()
  ```
  ```
      account	name	street	city	state	postal-code	Jan	Feb	Mar
  0	211829	Kerluke, Koepp and Hilpert	34456 Sean Highway	New Jaycob	Texas	28752	10000	62000	35000
  1	320563	Walter-Trantow	1311 Alvis Tunnel	Port Khadijah	NorthCarolina	38365	95000	45000	35000
  2	648336	Bashirian, Kunde and Price	62184 Schamberger Underpass Apt. 231	New Lilianland	Iowa	76517	91000	120000	35000
  3	109996	D'Amore, Gleichner and Bode	155 Fadel Crescent Apt. 144	Hyattburgh	Maine	46021	45000	120000	10000
  4	121213	Bauch-Goldner	7274 Marissa Common	Shanahanchester	California	49681	162000	120000	35000
  ```
  ```python
  df["account"].head(3)    # df[["account","name","street"]].head() 여러개의 columns 추출 가능
  ```
  ```
  0    211829
  1    320563
  2    648336
  Name: account, dtype: int64
  ```
  ```python
  df[:3]    # df["account"][:3] 원하는 columns 에서만도 가능
  ```
  ```
      account	name	street	city	state	postal-code	Jan	Feb	Mar
  0	211829	Kerluke, Koepp and Hilpert	34456 Sean Highway	New Jaycob	Texas	28752	10000	62000	35000
  1	320563	Walter-Trantow	1311 Alvis Tunnel	Port Khadijah	NorthCarolina	38365	95000	45000	35000
  2	648336	Bashirian, Kunde and Price	62184 Schamberger Underpass Apt. 231	New Lilianland	Iowa	76517	91000	120000	35000
  ```
  
  ### Series selection
  ```python
  account_series = df["account"]
  account_series[:3]
  ```
  ```
  0    211829
  1    320563
  2    648336
  Name: account, dtype: int64
  ```
  ```python
  account_series[[1,5,3]]
  ```
  ```
  1    320563
  5    132971
  3    109996
  Name: account, dtype: int64
  ```
  ```python
  account_series[account_series>250000]
  ```
  ```
  1     320563
  2     648336
  13    268755
  14    273274
  Name: account, dtype: int64
  ```
  
  ### index 변경
  ```python
  df.index = df["account"]
  del df["account"]
  df.head()
  ```
  ```
          name	street	city	state	postal-code	Jan	Feb	Mar
  account								
  211829	Kerluke, Koepp and Hilpert	34456 Sean Highway	New Jaycob	Texas	28752	10000	62000	35000
  320563	Walter-Trantow	1311 Alvis Tunnel	Port Khadijah	NorthCarolina	38365	95000	45000	35000
  648336	Bashirian, Kunde and Price	62184 Schamberger Underpass Apt. 231	New Lilianland	Iowa	76517	91000	120000	35000
  109996	D'Amore, Gleichner and Bode	155 Fadel Crescent Apt. 144	Hyattburgh	Maine	46021	45000	120000	10000
  121213	Bauch-Goldner	7274 Marissa Common	Shanahanchester	California	49681	162000	120000	35000
  ```
  
  ### Basic, loc, iloc selection
  ```python
  df[["name","street"]][:2]    # Column 과 index number
  df.loc[[211829,320563],["name","street"]]    # Column 과 index name
  df.iloc[:2,:2]    # Column number 와 index number
  ```
  ```
        	name	street
  account		
  211829	Kerluke, Koepp and Hilpert	34456 Sean Highway
  320563	Walter-Trantow	1311 Alvis Tunnel
  ```
  ```python
  df[["name","street"]].iloc[:5]
  ```
  ```
      name	street
  account		
  211829	Kerluke, Koepp and Hilpert	34456 Sean Highway
  320563	Walter-Trantow	1311 Alvis Tunnel
  648336	Bashirian, Kunde and Price	62184 Schamberger Underpass Apt. 231
  109996	D'Amore, Gleichner and Bode	155 Fadel Crescent Apt. 144
  121213	Bauch-Goldner	7274 Marissa Common
  ```
  
  ### index 재설정
  ```python
  df.index = list(range(0,15))
  ```
  ```python
  df.reset_index()
  ```
  ### Data Drop
  ```python
  df.drop([0,1,2,3])
  ```
  ```python
  df.drop("city", axis=1)    # df.drop(["city","state"], axis=1) 여러개 없애기
  ```
  
  [실행코드보기](./Selection&Drop.ipynb)

## lambda, map, apply
  ### lambda 함수
  - 한 줄로 함수를 표현하는 익명 함수 기법
  - Lisp 언어에서 시작된 기법으로 오늘날 현대언어에 많이 사용
  - `lambda argument : expression`
  ```python
  def f(x, y):
    return x + y
    
  f = lambda x,y : x + y
  ```
  ```python
  f = lambda x, y: x + y
  f(1,4)
  ```
  `5`
  ```python
  (lambda x: x + 1)(5)
  ```
  `6`
  
  ### map 함수
  - 함수와 sequence형 데이터를 인자로 받아
  - 각 element 마다 입력받은 함수를 적용하여 list로 변환
  - 일반적으로 함수를 lambda 형태로 표현함
  - `map(function, sequence)`
  ```python
  ex = [1,2,3,4,5]
  f = lambda x: x**2
  list(map(f,ex))
  ```
  `[1, 4, 9, 16, 25]`
  - 두 개 이상의 argument가 있을 때는 두 개의 sequence형을 써야함
  ```python
  f = lambda x, y : x + y
  list(map(f,ex,ex))
  ```
  `[2, 4, 6, 8, 10]`
  - 익명 함수 그대로 사용할 수 있음
  ```python
  list(map(lambda x: x+x, ex))
  ```
  `[2, 4, 6, 8, 10]`
  
  ### map for series
  - Pandas의 series type의 데이터에도 map 함수 사용가능
  - function 대신 dict, sequence형 자료등으로 대체 가능
  - series 단위에서 많이 
  ```python
  s1 = Series(np.arange(10))
  s1.head(5)
  ```
  ```
  0    0
  1    1
  2    2
  3    3
  4    4
  dtype: int32
  ```
  ```python
  s1.map(lambda x: x**2).head(5)
  ```
  ```
  0     0
  1     1
  2     4
  3     9
  4    16
  dtype: int64
  ```
  ```python
  z = {1:'A',2:'B',3:'C'}
  s1.map(z).head(5)    # dict type 으로 데이터 교체, 없는 값은 NaN
  ```
  ```
  0    NaN
  1      A
  2      B
  3      C
  4    NaN
  dtype: object
  ```
  ```python
  s2 = Series(np.arange(10,20))
  s1.map(s2).head(5)    # 같은 위치의 데이터를 s2로 전환
  ```
  ```
  0    10
  1    11
  2    12
  3    13
  4    14
  dtype: int3
  ```
  - Example - map for seires (성별 1,0 으로 변환)
  ```python
  df = pd.read_csv("./data/wages.csv")
  df.sex.unique()
  ```
  `array(['male', 'female'], dtype=object)`
  ```python
  df["sex_code"] = df.sex.map({"male":0,"female":1})
  df.head(5)
  ```
  ```
    earn	height	sex	race	ed	age	sex_code
  0	79571.299011	73.89	male	white	16	49	0
  1	96396.988643	66.23	female	white	16	62	1
  2	48710.666947	63.77	female	white	16	33	1
  3	80478.096153	63.22	female	other	16	95	1
  4	82089.345498	63.08	female	white	17	43	1
  ```
  - lambda 로 height 나누기
  ```python
  df["height_level"] = df.height.map(lambda x: 'L' if x > 70 else('M' if x > 60 else 'S'))
  ```
  
  ### Replace function
  - map 함수의 기능 중 데이터 변환 기능만 담당
  - 데이터 변환시 많이 사용하는 함수
  ```python
  df.sex.replace({"male":0, "female":1}).head()    # dict type 적용
  ```
  ```
  0    0
  1    1
  2    1
  3    1
  4    1
  Name: sex, dtype: int64
  ```
  ```python
  df.sex.replace(["male","female"],[0,1],inplace=True)    # Target list, Conversion list, inplace -> 데이터 변환결과를 적용
  df.head(5)
  ```
  ```
    earn	height	sex	race	ed	age	sex_code	height_level
  0	79571.299011	73.89	0	white	16	49	0	L
  1	96396.988643	66.23	1	white	16	62	1	M
  2	48710.666947	63.77	1	white	16	33	1	M
  3	80478.096153	63.22	1	other	16	95	1	M
  4	82089.345498	63.08	1	white	17	43	1	M
  ```
  
  ### apply for DataFrame
  - map 과 달리, series 전체(columns)에 해당 함수를 적용
  - 입력값이 series 데이터로 입력받아 handling 가능
  - 내장 연산 함수를 사용할 때도 똑같은 효과를 거둘 수 있음
  - mean, std 등 사용가능
  - scalar 값 이외에 series 값의 반환도 가능함
  ```python
  df_info = df[["earn","height","age"]]
  
  f = lambda x : x.max() - x.min()
  df_info.apply(f)
  ```
  ```
  earn      318047.708444
  height        19.870000
  age           73.000000
  dtype: float64
  ```
  ```python
  df_info.sum()      # df_info.apply(sum)
  ```
  ```
  earn      4.474344e+07
  height    9.183125e+04
  age       6.250800e+04
  dtype: float64
  ```
  ```python
  def f(x):
    return Series([x.min(), x.max()], index = ["min","max"])
  df_info.apply(f)
  ```
  ```
      earn	height	age
  min	-98.580489	57.34	22
  max	317949.127955	77.21	95
  ```
  
  ### applymap for DataFrame
  - series 단위가 아닌 element 단위로 함수를 적용함
  - series 단위에 apply를 적용시킬 때와 같은 효과
  ```python
  f = lambda x: -x
  df_info.applymap(f).head(5)
  ```
  ```
  earn	height	age
  0	-79571.299011	-73.89	-49
  1	-96396.988643	-66.23	-62
  2	-48710.666947	-63.77	-33
  3	-80478.096153	-63.22	-95
  4	-82089.345498	-63.08	-43
  ```
  ```python
  df_info["earn"].apply(f).head(5)
  ```
  ```
  0   -79571.299011
  1   -96396.988643
  2   -48710.666947
  3   -80478.096153
  4   -82089.345498
  Name: earn, dtype: float64
  ```
  
  [실행코드보기](./lambda,map,apply.ipynb)
  

## DataFrame Operations
  ### Series Operation
  ```python
  s1 = Series(range(1,6), index=list("abcde"))
  s2 = Series(range(5,11), index=list("bcedef"))
  
  s1 + s2    # s1.add(s2)
  ```
  ```
  a     NaN
  b     7.0
  c     9.0
  d    12.0
  e    12.0
  e    14.0
  f     NaN
  dtype: float64
  ```
  ```python
  s1.add(s2, fill_value=0)    # fill_value : 빈 값에 값 넣어주기
  ```
  ```
  a     1.0
  b     7.0
  c     9.0
  d    12.0
  e    12.0
  e    14.0
  f    10.0
  ```
  
  ### DataFrame Operation
  ```python
  df1 = DataFrame(np.arange(9).reshape(3,3),columns=list("abc"))
  df2 = DataFrame(np.arange(16).reshape(4,4),columns=list("abcd"))
  df1 + df2
  ```
  ```
    a	b	c	d
  0	0.0	2.0	4.0	NaN
  1	7.0	9.0	11.0	NaN
  2	14.0	16.0	18.0	NaN
  3	NaN	NaN	NaN	NaN
  ```
  ```python
  df1.add(df2, fill_value = 0)
  ```
  ```
    a	b	c	d
  0	0.0	2.0	4.0	3.0
  1	7.0	9.0	11.0	7.0
  2	14.0	16.0	18.0	11.0
  3	12.0	13.0	14.0	15.0
  ```
  
  ### Series + DataFrame
  ```python
  df = DataFrame(np.arange(16).reshape(4,4), columns=list("abcd"))
  df
  ```
  ```
    a	b	c	d
  0	0	1	2	3
  1	4	5	6	7
  2	8	9	10	11
  3	12	13	14	15
  ```
  ```python
  s = Series(np.arange(10,14),index=list("abcd"))
  s
  ```
  ```
    a    10
  b    11
  c    12
  d    13
  dtype: int32
  ```
  ```python
  df + s    # column을 기준으로 broadcasting이 발생함
  ```
  ```
      a	b	c	d
  0	10	12	14	16
  1	14	16	18	20
  2	18	20	22	24
  3	22	24	26	28
  ```
  ```python
  s2 = Series(np.arange(10,14))
  s2
  ```
  ```
  0    10
  1    11
  2    12
  3    13
  dtype: int32
  ```
  ```python
  df + s2   # series index 와 DataFrame의 columns 이름이 매칭
  ```
  ```
    a	b	c	d	0	1	2	3
  0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  1	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  2	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  3	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  ```
  ```python
  df.add(s2,axis=0)
  ```
  ```
    a	b	c	d
  0	10	11	12	13
  1	15	16	17	18
  2	20	21	22	23
  3	25	26	27	28
  ```
  
  [실행코드보기](./DataFrame_Operations.ipynb)
  
 
## Pandas builit-in functions
  ### describe
  - Numeric type 데이터의 요약 정보를 보여줌
  ```python
  df = pd.read_csv("./data/wages.csv")
  df.describe()
  ```
  ```
      earn	height	ed	age
  count	1379.000000	1379.000000	1379.000000	1379.000000
  mean	32446.292622	66.592640	13.354605	45.328499
  std	31257.070006	3.818108	2.438741	15.789715
  min	-98.580489	57.340000	3.000000	22.000000
  25%	10538.790721	63.720000	12.000000	33.000000
  50%	26877.870178	66.050000	13.000000	42.000000
  75%	44506.215336	69.315000	15.000000	55.000000
  max	317949.127955	77.210000	18.000000	95.000000
  ```
  
  ### unique
  - series data의 유일한 값을 list로 반환함
  ```python
  df.race.unique()
  ```
  `array(['white', 'other', 'hispanic', 'black'], dtype=object)`
  ```python
  np.array(dict(enumerate(df["race"].unique())))    # dict type 으로 index
  ```
  `array({0: 'white', 1: 'other', 2: 'hispanic', 3: 'black'}, dtype=object)`
  ```python
  value = list(map(int, np.array(list(enumerate(df['race'].unique())))[:,0].tolist()))
  key = np.array(list(enumerate(df['race'].unique())),dtype=str)[:,1].tolist()

  value, key
  ```
  `([0, 1, 2, 3], ['white', 'other', 'hispanic', 'black'])`
  ```python
  df["race"].replace(to_replace=key, value=value, inplace=True)
  
  value = list(map(int, np.array(list(enumerate(df['sex'].unique())))[:,0].tolist()))
  key = np.array(list(enumerate(df['sex'].unique())),dtype=str)[:,1].tolist()

  value, key
  ```
  `([0, 1], ['male', 'female'])`
  ```python
  df["sex"].replace(to_replace=key, value=value, inplace=True)
  df.head(5)
  ```
  ```
    earn	height	sex	race	ed	age
  0	79571.299011	73.89	0	0	16	49
  1	96396.988643	66.23	1	0	16	62
  2	48710.666947	63.77	1	0	16	33
  3	80478.096153	63.22	1	1	16	95
  4	82089.345498	63.08	1	0	17	43
  ```
  
  ### Sum
  - 기본적인 column 또는 row 값의 연산을 지원
  - sub, mean, min, max, count, median, mad, var 등
  ```python
  df.sum(axis=0)   # column 별
  df.sum(axis=1)    # row 별
  ```
  
  ### isnull
  - column 또는 row 값의 NaN(null) 값의 index를 반환함
  ```python
  df.isnull().head()
  ```
  ```
    earn	height	sex	race	ed	age
  0	False	False	False	False	False	False
  1	False	False	False	False	False	False
  2	False	False	False	False	False	False
  3	False	False	False	False	False	False
  4	False	False	False	False	False	False
  ```
  ```python
  df.isnull().sum()
  ```
  ```
  earn      0
  height    0
  sex       0
  race      0
  ed        0
  age       0
  dtype: int64
  ```
  
  ### sort_values
  - column 값을 기준으로 데이터를 sorting
  ```python
  df.sort_values(["age","earn"], ascending=True).head(10)
  ```
  ```
        earn	height	sex	race	ed	 age
  1038	-56.321979	67.81	male	2	10	22
  800	-27.876819	72.29	male	0	12	22
  963	-25.655260	68.90	male	0	12	22
  1105	988.565070	64.71	female	0	12	22
  801	1000.221504	64.09	female	0	12	22
  862	1002.023843	66.59	female	0	12	22
  933	1007.994941	68.26	female	0	12	22
  988	1578.542814	64.53	male	0	12	22
  522	1955.168187	69.87	female	3	12	22
  765	2581.870402	64.79	female	0	12	22
  ```
  
  ### etc
  - df.cumsum()
  - df.cummax()
  
  ### Correlation & Covariance
  - 상관계수와 공분산을 구하는 함수
  - corr, cov, corrwith
  ```python
  df.age.corr(df.earn)
  ```
  `0.07400349177836055`
  ```python
  df.age.cov(df.earn)
  ```
  `36523.69921040889`
  ```python
  df.corrwith(df.earn)
  ```
  ```
  earn      1.000000
  height    0.291600
  race     -0.063977
  ed        0.350374
  age       0.074003
  dtype: float64
  ```
  ```python
  df.corr()
  ```
  ```
        earn	height	race	ed	age
  earn	1.000000	0.291600	-0.063977	0.350374	0.074003
  height	0.291600	1.000000	-0.045974	0.114047	-0.133727
  race	-0.063977	-0.045974	1.000000	-0.049487	-0.056879
  ed	0.350374	0.114047	-0.049487	1.000000	-0.129802
  age	0.074003	-0.133727	-0.056879	-0.129802	1.000000
  ```
  
  [실행코드보기](./built-in_functions.ipynb)


## Groupby1
  ### Groupby
  - SQL groupby 명령어와 같음
  - split -> apply -> combine
  - 과정을 거쳐서 연산함
  - `df.groupby("Team")["Points"].sum()`
  - ("Team") - 묶음의 기준이 되는 column
  - ["Points"] - 적용받는 column
  - .sum() - 적용받는 연산
  ```python
  ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}

  df = pd.DataFrame(ipl_data)
  df
  ```
  ```
      Team	Rank	Year	Points
  0	Riders	1	2014	876
  1	Riders	2	2015	789
  2	Devils	2	2014	863
  3	Devils	3	2015	673
  4	Kings	3	2014	741
  5	kings	4	2015	812
  6	Kings	1	2016	756
  7	Kings	1	2017	788
  8	Riders	2	2016	694
  9	Royals	4	2014	701
  10	Royals	1	2015	804
  11	Riders	2	2017	690
  ```
  ```python
  df.groupby("Team")["Points"].sum()
  ```
  ```
    Team
  Devils    1536
  Kings     2285
  Riders    3049
  Royals    1505
  kings      812
  Name: Points, dtype: int64
  ```
  - 한 개 이상의 column을 묶을 수 있음
  ```python
  df.groupby(["Team","Year"])["Points"].sum()
  ```
  ```
    Team    Year
  Devils  2014    863
          2015    673
  Kings   2014    741
          2016    756
          2017    788
  Riders  2014    876
          2015    789
          2016    694
          2017    690
  Royals  2014    701
          2015    804
  kings   2015    812
  Name: Points, dtype: int64
  ```
  ### Hierarchical index
  - Groupby 명령의 결과물도 결국은 dataframe
  - 두 개의 column 으로 groupby를 할 경우, index가 두 개 생성
  - `h_index.index`
  #### Hierarchical index - unstack()
  - Group으로 묶여진 데이터를 matrix 형태로 전환해줌
  - `h_index.unstack()`
  #### Hierarchical index - swaplevel
  - index level을 변경할 수 있음
  - `h_index.swaplevel()`
  #### Hierarchical index - operations
  - index level 을 기준으로 기본 연산 수행 가능
  - `h_index.sum(level=0)`
  - `h_index.sum(level=1)`
  ```python
  h_index = df.groupby(["Team","Year"])["Points"].sum()
  h_index
  ```
  ```
    Team    Year
  Devils  2014    863
          2015    673
  Kings   2014    741
          2016    756
          2017    788
  Riders  2014    876
          2015    789
          2016    694
          2017    690
  Royals  2014    701
          2015    804
  kings   2015    812
  Name: Points, dtype: int64
  ```
  ```python
  h_index.unstack()    # h_index.unstack().fillna(0)  NaN 값 0
  ```
  ```
    Year	2014	2015	2016	2017
  Team				
  Devils	863.0	673.0	NaN	NaN
  Kings	741.0	NaN	756.0	788.0
  Riders	876.0	789.0	694.0	690.0
  Royals	701.0	804.0	NaN	NaN
  kings	NaN	812.0	NaN	NaN
  ```
  
  [실행코드보기](./groupby1.ipynb)


## Groupby2
  ### Grouped
  - Groupby에 의해 split 된 상태를 추출 가능함
  ```python
  grouped = df.groupby("Team")
  for name, group in grouped:
    print(name)
    print(group)
  ```
  ```
    Devils
       Team  Rank  Year  Points
  2  Devils     2  2014     863
  3  Devils     3  2015     673
  Kings
      Team  Rank  Year  Points
  4  Kings     3  2014     741
  6  Kings     1  2016     756
  7  Kings     1  2017     788
  Riders
        Team  Rank  Year  Points
  0   Riders     1  2014     876
  1   Riders     2  2015     789
  8   Riders     2  2016     694
  11  Riders     2  2017     690
  Royals
        Team  Rank  Year  Points
  9   Royals     4  2014     701
  10  Royals     1  2015     804
  kings
      Team  Rank  Year  Points
  5  kings     4  2015     812
  ```
  - 특정 key 값을 가진 그룹의 정보만 추출 가능
  ```python
  grouped.get_group("Devils")
  ```
  ```
    Team	Rank	Year	Points
  2	Devils	2	2014	863
  3	Devils	3	2015	673
  ```
  - 추출된 group 정보에는 세 가지 유형의 apply 가 가능함
  - Aggregation : 요약된 통계정보를 추출해 줌
  - Transformation : 해당 정보를 변환해줌
  - Filtration : 특정 정보를 제거하여 보여주는 필터링 기능
  #### Aggregation
  ```python
  grouped.agg(sum)
  ```
  ```
      Rank	Year	Points
  Team			
  Devils	5	4029	1536
  Kings	5	6047	2285
  Riders	7	8062	3049
  Royals	5	4029	1505
  kings	4	2015	812
  ```
  ```python
  grouped.agg(np.mean)
  ```
  ```
      Rank	Year	Points
  Team			
  Devils	2.500000	2014.500000	768.000000
  Kings	1.666667	2015.666667	761.666667
  Riders	1.750000	2015.500000	762.250000
  Royals	2.500000	2014.500000	752.500000
  kings	4.000000	2015.000000	812.000000
  ```
  - 특정 column 에 여러개의 function을 Apply 할 수도 있음
  ```python
  grouped['Points'].agg([np.sum,np.mean,np.std])
  ```
  ```
      sum	mean	std
  Team			
  Devils	1536	768.000000	134.350288
  Kings	2285	761.666667	24.006943
  Riders	3049	762.250000	88.567771
  Royals	1505	752.500000	72.831998
  kings	812	812.000000	NaN
  ```
  
  #### Transformation
  - Aggregation 과 달리 key 값 별로 요약된 정보가 아님
  - 개별 데이터의 변환을 지원함
  ```python
  score = lambda x: (x.max())
  grouped.transform(score)
  ```
  ```python
  score = lambda x: (x - x.mean()) / x.std()
  grouped.transform(score)
  ```
  #### Filter
  - 특정 조건으로 데이터를 검색할 때 사용
  - `df.groupby('Team').filter(lambda x: len(x) >= 3)`
  - filter 안에는 boolean 조건이 존재해야함
  - len(x)는 grouped 된 dataframe 개수
  - `df.groupby('Team').filter(lambda x: x["Rank"].sum() >= 2)`
  - `df.groupby('Team').filter(lambda x: x["Rank"].mean() >= 1)`

  [실행코드보기](./groupby2.ipynb)


## Pivot table & Crosstab
  ### Pivot Table
  - 우리가 Excel 에서 보던 그것!
  - index 축은 gropuby와 동일함
  - column 에 추가로 labelling 값을 추가하여, Value에 numeric type 값을 aggregation 하는 형태
  ```python
  df_phone.pivot_table(['duration'], index=[df_phone.month, df_phone.item],
                    columns=df_phone.network, aggfunc="sum", fill_value=0)
  ```
  
  ### Crosstab
  - 특히 두 칼럼에 교차 빈도, 비율, 덧셈 등을 구할 때 사용
  - Pivot table 의 특수한 형태
  - User-Item Rating Matrix 등을 만들 때 사용 가능함
  ```python
  pd.crosstab(index=df_movie.critic, columns=df_movie.title, values=df_movie.rating,
           aggfunc="first").fillna(0)
  ```
  - gropuby로도 만들 수 있음!
  ```python
  df_movie.groupby(['critic','title']).agg({"rating":"sum"}).unstack().fillna(0)
  ```
  
  [실행코드보기](./pivot_table&crosstab.ipynb)


## Merge & Concat
  ### Merge
  - SQL 에서 많이 사용하는 Merge 와 같은 기능
  - 두 개의 데이터를 하나로 합침
  - `pd.merge(df_a, df_b, on='subject_id')`
  - 두 dataframe의 column 이름이 다를 때
  - `pd.merge(df_a, df_b, left_on='subject_id', right_on='subject_id')`
  #### join method
  - left - 왼쪽 정보 다 나옴
  - `pd.merge(df_a, df_b, on='subject_id', how='left')`
  - right - 오른쪽 정보 다 나옴
  - `pd.merge(df_a, df_b, on='subject_id', how='right')`
  - outer - 양쪽
  - `pd.merge(df_a, df_b, on='subject_id', how='outer')`
  - inner - 둘 다 있을 경우
  - `pd.merge(df_a, df_b, on='subject_id', how='inner')`

  ### index based join
  - index 번호를 기준으로 join
  - `pd.merge(df_a, df_b, right_index=True, left_index=True)`

  ### Concat
  - 같은 형태의 데이터를 붙이는 연산작업
  - 기본적으로 세로로 합쳐짐
  ```python
  df_new = pd.concat([df_a, df_b])    # == df_a.append(df_b)
  df_new.reset_index()   
  ```
  ```python
  df_new = pd.concat([df_a, df_b], axis=1)   # 가로로 붙이기
  df_new.reset_index()
  ```
  
  [실행코드보기](./merge&concat.ipynb)
  

## Database connection & Persistance
  ### Database connection
  - Data loading 시 db connection 기능을 제공함
  
  ### XLS persistence
  - DataFrame의 엑셀 추출 코드
  - Xls 엔진으로 openpyxls 또는 XlsxWrite 사용
  
  ### Pickle persistence
  - 가장 일반적인 python 파일 persistence
  - to_pickle, read_pickle 함수 이용
  - `df_routes.to_pickle("./data/df_routes.pickle")`
  - `df_routes_pickle = pd.read_pickle("./data/df_routes.pickle")`

  [실행코드보기](./database_connection&persistence.ipynb)
