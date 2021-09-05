# Data Preprocessing
  ## 데이터 처리의 전략
  - 모판은 흔들지 않는다
  - 하나의 셀은 다시 실행해도 그 결과가 보장되야 한다
  - 전처리가 완료후 함수화한다 (merge 함수 필수)
  - 컬럼 이름은 list로 관리하기! 직접 입력 X
  - 데이터는 타입별로 분리해서 관리하기!
  - **데이터 노트 작성하기!!**
    ### 데이터 노트
    - 데이터에 대한 처리 내용 및 방향을 정리한 노트
    - 기본적인 전처리 방향과 방법들을 정리함
    - 데이터에 대한 아이디어를 정리와 지속적인 업데이트
    - 기본적인 데이터 현황 파악 코드
    ```python
    df.dtypes
    df.info()
    df.isnull().sum()
    df.describe()
    df.head(2).T
    ```
    ### Data Cleansing issues
    - 데이터가 빠진 경우 (결측치의 처리)
    - 라벨링된 데이터 (category) 데이터의 처리
    - 데이터의 scale의 차이가 매우 크게 날 경우
 
  ## Missing Value Strategy
  - 데이터가 없으면 sample을 drop
  - 데이터가 없는 최소 개수를 정해서 sample을 drop
  - 데이터가 거의 없는 feature는 feature 자체를 drop
  - 최빈값, 평균값으로 비어있는 데이터를 추가

    ### Data Fill
    ```python
    fillna_df["preTestScore"] = cleaned_df["preTestScore"].fillna(cleaned_df["preTestScore"].mean())
    fillna_df["preTestScore"]
    fillna_df["postTestScore"] = cleaned_df["postTestScore"].fillna(cleaned_df.groupby("sex")["postTestScore"].transform("mean"))
    ```
    
  ## 이산형 데이터를 어떻게 처리할까?
  **One - Hot Encoding**
  - `pd.get_dummies()`
  ```python
  pd.get_dummies(edges[["color"]])
  ```
  ![image](https://user-images.githubusercontent.com/76936390/115048525-9e5d2980-9f14-11eb-9825-81a4bbaf3369.png)
  ```python
  pd.merge(edges, pd.get_dummies(edges[["color"]]), left_index=True, right_index=True)
  ```
  ![image](https://user-images.githubusercontent.com/76936390/115048602-b2a12680-9f14-11eb-91c9-2bdbfccd1f36.png)

  **Data binnig**
  ```python
  bins = [0, 50, 80, 100] # Define bins as 0 to 25, 25 to 50, 60 to 75, 75 to 100
  group_names = ['Low', 'Good', 'Great']
  categories = pd.cut(df['postTestScore'], bins, labels=group_names)
  ```
  
  ## Feature Scaling
  - Feature 간의 최대-최소값의 차이를 맞춘다
  - sklearn을 주로 사용
    ### Min-Max Normalization
    - 기존 변수에 범위를 새로운 최대-최소로 변경
    - 일반적으로 0~1 사이 값으로 변경함
    
    ### Standardization 
    - 기존 변수에 범위를 정규분포로 변환
    
    
    ```python
    from sklearn import preprocessing
    std_scaler = preprocessing.StandardScaler().fit(df[['Alcohol', 'Malic acid']])
    df_std = std_scaler.transform(df[['Alcohol', 'Malic acid']])
    
    minmax_scaler = preprocessing.MinMaxScaler().fit(df[['Alcohol', 'Malic acid']])
    minmax_scaler.transform(df[['Alcohol', 'Malic acid']])
    ```
  ## Feature Engineering
  - 가장 적합한 특성을 찾는것
    ### Generation
    - Binarization, Quantization
    - Scaling (normalization)
    - Interaction features
    - Log transformation
    - Dimension reduction
    - Clustering
    
    ### Selection
    - Univariate statics
    - Model-based selection
    - Iterative feature selection
    - Feature removal

    ### Log transformations
    - 데이터의 분포가 극단적으로 모였을 때(poisson)
    - 선형 모델은 데이터가 정규분포때 적합
    - Poisson -> Normal distribution
    - `np.log` or `np.exp` 등의 함수를 사용
    
    ### Mean encoding
    - Category 데이터는 항상 One-hot Encoding?
    - -> X, 다양한 인코딩 기법이 있음
    - 대표적인 방법으로 Y값에 대한 분포를 활용한 Mean Encoding이 사용됨
    - Label 인코딩은 그 자체로 정보가 존재하지 않음
    - Mean 인코딩 : 분포의 값을 취할 수 있음
    
    ### Interaction features
    - 기존 feature 들의 조합으로 새로운 feature를 생성
    - Data에 대한 사전 지식과 이해가 필요
    - Polynomial feature를 사용한 자동화 가능 
    
    ### Feature selection
    - 모든 feature 들이 반드시 model 학습에 필요치 않다
    - 어떤 feature 들은 성능을 오히려 나쁘게 함
    - 너무 많은 feature -> overfitting의 원인
    - 모델에 따라서 필요한 feature를 선택함
    - 필요없는 feature 제거 -> 학습 속도와 성능 향상
    - 다양한 기법과 코드에 대해 공부
    
    #### feature 선택의 주의 사항들
    - prediction time 에도 쓸 수 있는 feature 인가?
    - 실시간 예측이 필요할 때, 생성이 너무 고비용이 아닌가?
    - scale은 일정한가? 또는 비율적으로 표현 가능한가?
    - 새롭게 등장하는 category data는? 가장 비슷한 것
    - 너무 극단적인 분포 -> threshold 기반으로 잘라내기
    
    #### 이런 feature 들은 삭제하자!
    - Correlation 이 너무 높은 feature는 삭제
    - 전처리가 완료된 str feature들
    - ID와 같은 성향을 가진 feature 들
