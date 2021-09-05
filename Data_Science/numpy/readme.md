# About Numpy
## numpy
* numpy는 np.array 함수를 활용하여 배열을 생성함
* numpy는 하나의 데이터 type 만 배열에 넣을 수 있음
* list 와 가장 큰 차이점, Dynamic typing not supported
* C의 Array를 사용하여 배열을 생성함
  ```python
  test_array = np.array([1, 4, 5, "8"], float)  #String type의 데이터를 넣어도 뒤에서 선언해주는 type에 맞춰서 생성
  test_array
  ```
  `array([1., 4., 5., 8.])`
  ```python
  type(test_array[3])
  ```
  `numpy.float64`


  ### - shape : numpy array의 object의 dimension 구성을 반환함
  ### - dtype : numpy array의 데이터 type을 반환함
    ```python
    matrix = [[1,2,5,8],[1,2,5,8],[1,2,5,8]]
    matrix = np.array(matrix, int)
    tensor = [[[1,2,5,8],[1,2,5,8],[1,2,5,8]],[[1,2,5,8],[1,2,5,8],[1,2,5,8]],[[1,2,5,8],[1,2,5,8],[1,2,5,8]]]
    tensor = np.array(tensor, int)
    ```
    ```python
    test_array.shape
    ```
    `(4,)`
    ```python
    test_array.dtype
    ```
    `dtype('float64')`

    ```python
    matrix.shape
    ```
    `(3, 4)`
    ```python
    tensor.shape
    ```
    `(3, 3, 4)`


  ### - ndim : number of dimension
  ### - size : data의 개수

    ```python
    matrix.ndim
    ```
    `2`
    ```python
    tensor.ndim
    ```
    `3`
    ```python
    matrix.size
    ```
    `12`
    ```python
    tensor.ndim
    ```
    `36`


  ### - reshape 
    - Array의 shape의 크리를 변경함 (element의 갯수는 동일)
    - Array의 size만 같다면 다차원으로 자유로이 변형 가능

    ```python
    test_matrix = [[1,2,3,4], [1,2,5,8]]
    np.array(test_matrix).shape
    ```
    `(2, 4)`
    ```python
    np.array(test_matrix).reshape(8,)
    ```
    `array([1, 2, 3, 4, 1, 2, 5, 8])`
    ```python
    np.array(test_matrix).reshape(8,).shape
    ```
    `(8,)`
    ```python
    np.array(test_matrix).reshape(-1,2).shape    # -1 : size를 기반으로 row 개수 선정
    ```
    `(4, 2)`
    ```python
    np.array(test_matrix).reshape(2,2,2)
    ```
    ```
    array([[[1, 2],
          [3, 4]],

         [[1, 2],
          [5, 8]]])
    ```
    ```python
    np.array(test_matrix).reshape(2,2,2).shape
    ```
    `(2, 2, 2)`


  ### - flatten
    - 다차원 array를 1차원 array로 변환

    ```python
    test_matrix = [[[1,2,3,4],[1,2,5,7]], [[1,2,3,4],[1,2,5,8]]]
    np.array(test_matrix).flatten()
    ```
    `array([1, 2, 3, 4, 1, 2, 5, 7, 1, 2, 3, 4, 1, 2, 5, 8])`

    [실행 코드 보기](./ndarray,reshape.ipynb)

## indexing & slicing
  ### - indexing
   - list와 달리 이차원 배열에서 [0,0] 과 같은 표기법을 제공
   - Matrix 일 경우 앞은 row 뒤는 column을 의미함

  ```python
  test_example = np.array([[1,2,3], [4.5,5,6]], int)
  test_exampleple
  ```
  ```
  array([[1, 2, 3],
       [4, 5, 6]])
  ```
  ```python
  test_example[0][0]
  ```
  `1`
  ```python
  test_example[0,0]
  ```
  `1`
  ```python
  test_example[0,2] = 10    # Matrix 0,2 에 10 할당
  test_example
  ```
  ```
  array([[ 1,  2, 10],
       [ 4,  5,  6]])
  ```


  ### - slicing
   - list 와 달리 행과 열 부분을 나눠서 slicing이 가능함
   - Matrix 의 부분 집합을 추출할 때 유용함

  ```python
  a = np.array([[1,2,3,4,5],[6,7,8,9,10]],int)
  a[:,2:]    # 전체 Row의 2열 이상
  ```
  ```
  array([[ 3,  4,  5],
       [ 8,  9, 10]])
  ```
  ```python
  a[1,1:3]    # 1 Row의 1열 ~ 2열
  ```
  `array([7, 8])`
  ```python
  a[1:3]    # 1 Row ~ 2 Row의 전체
  ```
  `array([[ 6,  7,  8,  9, 10]])`

  [실행 코드 보기](./indexing,slicing.ipynb)

## Create Functinos
  ### - arange
   - array의 범위를 지정하여, 값의 list를 생성하는 명령어

  ```python
  np.arange(10)
  ```
  `array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])`
  ```python
  np.arange(0, 5, 0.5)    # (시작, 끝, step) step에 floating point도 표시가능
  ```
  `array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])`
  ```python
  np.arange(30).reshape(5,6)
  ```
  ```
  array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29]])
  ```
  
  ### - zeros, ones and empty
   * zeros
    - 0으로 가득찬 ndarray 생성

      ```python
      np.zeros(shape=(10,), dtype=np.int8)    # 10 - zero vector 생성
      ```
      `array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int8)`
      ```python
      np.zeros((2,5))    # 2 by 5 - zero matrix 생성
      ```
      ```
      array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])
      ```
  * ones
    - 1로 가득찬 ndarray 생성
    - np.ones(shape, dtype, order)

    ```python
    np.ones(shape=(10,), dtype=np.int8)
    ```
    `array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int8)`
    ```python
    np.ones((2,5))
    ```
    ```
    array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])
    ```
  
  * empty
    - shape만 주어지고 비어있는 ndarray 생성
    - memory initialization 이 되지 않음

    ```python
    np.empty(shape=(10,), dtype=np.int8)
    ```
    `array([ 97,   0, 103,   0, 101,   0, 115,   0,  92,   0], dtype=int8)`
    ```python
    np.empty((3,5))
    ```
    ```
    array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])
    ```
    
  ### something_like
   - 기존 ndarray의 shape 크기 만큼 1, 0 또는 empty array 를 반환
   
   ```python
   test_matrix = np.arange(30).reshape(5,6)
   np.ones_like(test_matrix)
   ```
   ```
   array([[1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1]])
   ```
    
  ### identity
   - 단위 행렬(i 행렬)을 생성함

    ```python
    np.identity(n=3, dtype=np.int8)
    ```
    ```
    array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]], dtype=int8)
    ```
    ```python
    np.identity(5)
    ```
    ```
    array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
    ```

  ### eye
   - 대각선이 1인 행렬, k값의 시작 indext의 변경이 가능
      ```python
      np.eye(N=3, M=5, dtype=np.int8)
      ```
      ```
      array([[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0]], dtype=int8)
      ```
      ```python
      np.eye(3)
      ```
      ```
      array([[1., 0., 0.],
         [0., 1., 0.],
         [0., 0., 1.]])
      ```
      ```python
      np.eye(3,5,k=2)    # k -> start index
      ```
      ```
      array([[0., 0., 1., 0., 0.],
         [0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 1.]])
      ```
  ### diag
   - 대각 행렬의 값을 추출함
      ```python
      matrix = np.arange(9).reshape(3,3)
      np.diag(matrix)
      ```
      `array([0, 4, 8])`
      ```python
      np.diag(matrix, k=1)
      ```
      `array([1, 5])`
      
      
  ### random sampling
   - 데이터 분포에 따른 sampling으로 array를 생성
      ```python
      np.random.uniform(0,1,10).reshape(2,5)    # 균등분포
      ```
      ```
      array([[0.33626064, 0.49832527, 0.3860534 , 0.22872582, 0.62911835],
         [0.0313587 , 0.68151892, 0.58590359, 0.03118958, 0.8927543 ]])
      ```
      ```python
      np.random.normal(0,1,10).reshape(2,5)    # 정규분포
      ```
      ```
      array([[-0.63628088,  0.37574358, -0.22370008,  1.26157386, -0.48751402],
         [-1.03332467, -1.94924472, -0.00409418,  0.62437485, -2.6839599 ]])
      ```
      
      [실행 코드 보기](./creation_functions.ipynb)


## Operation Functions
  ### sum
   - ndarray의 element들 간의 합을 구함, list의 sum 기능과 동일
      ```python
      test_array = np.arange(1,11)
      test_array
      ```
      `array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])`
      ```python
      test_array.sum(dtype=np.float)
      ```
      `55.0`
   ### axis
   - 모든 operation function을 실행할 때 기준이 되는 dimension 축
   
   
   <img src="https://user-images.githubusercontent.com/76936390/111059842-da4b3c00-84db-11eb-9c78-d96b53029f8c.png" width="30%"> <img src="https://user-images.githubusercontent.com/76936390/111059849-eb944880-84db-11eb-9975-b3b475f3c569.png" width="35%">


   ### mean & std
   - ndarray의 element들 간의 평균 또는 표준 편차를 반환
   ```python
   test_array.mean(), test_array.mean(axis=0)
   ```
   `(6.5, array([5., 6., 7., 8.]))`
   ```python
   test_array.std(), test_array.std(axis=0)
   ```
   `(3.452052529534663, array([3.26598632, 3.26598632, 3.26598632, 3.26598632]))`
   
   ### concatenate
   - numpy array를 합치는 함수
   - vstack : 세로로 합치기
   - hstack : 가로로 합치기
   - concatenate : axis=0 -> vstack, axis=1 -> hstack 
   ```python
   a = np.array([1,2,3])
   b = np.array([5,7,8])
   c = np.vstack((a,b))
   c
   ```
   ```
   array([[1, 2, 3],
       [5, 7, 8]])
   ```
   ```python
   a = np.array([[1],[2],[3]])
   b = np.array([[5],[7],[8]])
   c = np.hstack((a,b))
   c
   ```
   ```
   array([[1, 5],
       [2, 7],
       [3, 8]])
   ```
   ```python
   a = np.array([[1, 2, 3]])
   b = np.array([[5, 7, 8]])
   c = np.concatenate((a,b), axis=0)
   c
   ```
   ```
   array([[1, 2, 3],
       [5, 7, 8]])
   ```
   ```python
   a = np.array([[1, 2], [3, 4]])
   b = np.array([[5, 6]])
   c = np.concatenate((a,b.T), axis=1)
   c
   ```
   ```
   array([[1, 2, 5],
       [3, 4, 6]])
   ```
   [실행 코드 보기](./operation_functions.ipynb)

  ## Array Operations
  - Numpy는 array 간의 기본적인 사칙 연산을 지원함
  ### Element-wise operations 
  - Array 간 shape이 같을 때 일어나는 연산 (같은 위치에 있는 값들끼리 연산)
  ```python
  test_a = np.array([[1,2,3],[4,5,6]])
  test_a + test_a
  ```
  ```
  array([[ 2,  4,  6],
       [ 8, 10, 12]])
  ```
  ```python
  test_a - test_a
  ```
  ```
  array([[0, 0, 0],
       [0, 0, 0]])
  ```
  ```python
  test_a * test_a
  ```
  ```
  array([[ 1,  4,  9],
       [16, 25, 36]])
  ```
  ### Dot product
  - Matrix의 기본 연산
  - dot 함수 사용
  ```python
  a = np.arange(1,7).reshape(2,3)
  b = np.arange(7,13).reshape(3,2)
  a.dot(b)
  ```
  ```
  array([[ 58,  64],
       [139, 154]])
  ```
  
  ### transpose
  - transpose 또는 T attribute 사용
  ```python
  a
  ```
  ```
  array([[1, 2, 3],
       [4, 5, 6]])
  ```
  ```python
  a.T # or a.transpose()
  ```
  ```
  array([[1, 4],
       [2, 5],
       [3, 6]])
  ```
  
  ### broadcasting
  - Shape 이 다른 배열 간 연산을 자동으로 지원하는 기능
  - Scalar - vector 외에도 vector - matrix 간의 연산도 지원
  ```python
  a + 3
  ```
  ```
  array([[4, 5, 6],
       [7, 8, 9]])
  ```
  ```python
  a - 2, a * 4, a / 2, a // 0.2, a ** 3 등등
  ```
  
  ```python
  test_matrix = np.arange(1,13).reshape(4,3)
  test_vector = np.arange(10,40,10)
  test_matrix + test_vector
  ```
  ```
  array([[11, 22, 33],
       [14, 25, 36],
       [17, 28, 39],
       [20, 31, 42]])
  ```
  
  [실행 코드 보기](./array_operations.ipynb)


 ## comparisions
  ### All & Any
  - Array 의 데이터 전부(and) 또는 일부(or)가 조건에 만족 여부 반환
  ```python
  a = np.arange(10)
  np.all(a>5), np.all(a<10)
  ```
  `(False, True)`
  ```python
  np.any(a>5), np.any(a<0)
  ```
  `(True, False)`
  ```python
  a<0
  ```
  `array([False, False, False, False, False, False, False, False, False, False])`
  
  - Numpy 는 배열의 크기가 동일 할 때 element간 비교의 결과를 Boolean type으로 반환하여 돌려줌
  ```python
  test_a = np.array([1, 3, 0], float)
  test_b = np.array([5, 2, 1], float)
  test_a > test_b
  ```
  `array([False,  True, False])`
  ```python
  test_a == test_b
  ```
  `array([False, False, False])`
  - logical_and, logical_not, logical_or 이라는 것도 있음

  ### np.where (중요)
  - where(condition, TRUE, FALSE)
  ```python
  np.where(test_a > 0, 3, 2)
  ```
  `array([3, 3, 2])`
  ```python
  np.where(a > 5)
  ```
  `(array([6, 7, 8, 9], dtype=int64),)`
  ```python
  a = np.array([1,np.NaN, np.Inf], float)
  np.isnan(a)
  ```
  `array([False,  True, False])`
  ```python
  np.isfinite(a)
  ```
  `array([ True, False, False])`
  
  ### argmax & argmin  (중요)
  - array 내 최대값 또는 최소값의 index를 반환함
  - axis 기반의 반환
  ```python
  a = np.array([1,2,4,5,8,78,23,3])
  np.argmax(a), np.argmin(a)
  ```
  `(5, 0)`
  ```python
  a = np.array([[1,2,4,7], [9,88,6,45],[9,76,3,4]])
  np.argmax(a, axis=1), np.argmin(a, axis=0)
  ```
  `(array([3, 1, 1], dtype=int64), array([0, 0, 2, 2], dtype=int64))`
  
  [실행 코드 보기](./comparisions.ipynb)


 ## boolean & fancy index
  ### boolean index
  - numpy는 배열의 특정 조건에 따른 값을 배열 형태로 추출 할 수 있음
  - Comparision operation 함수들도 모두 사용가능
  ```python
  a = np.array([1,4,0,2,3,8,9,7])
  a > 3
  ```
  `array([False,  True, False, False, False,  True,  True,  True])`
  ```python
  a[a > 3]
  ```
  `array([4, 8, 9, 7])`
  ```python
  condition = a < 3
  a[condition]
  ```
  `array([1, 0, 2])`
  ```
  A = np.random.randint(1,21, size=(9,9))
  B = A < 10
  B.astype(np.int)
  ```
  ```
  array([[0, 1, 0, 1, 1, 0, 1, 0, 0],
       [0, 0, 0, 1, 1, 0, 1, 0, 1],
       [1, 0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 1, 0],
       [1, 0, 0, 1, 0, 0, 1, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 0, 1],
       [0, 0, 0, 1, 0, 0, 1, 1, 0],
       [1, 1, 0, 1, 1, 1, 0, 0, 1],
       [0, 0, 0, 0, 0, 1, 0, 1, 0]])
  ```
   
  ### fancy index
  - numpy는 array를 index value로 사용해서 값을 추출하는 방법
  ```python
  a = np.array([2, 4, 6, 8])
  b = np.array([0, 0 ,1, 3, 2, 1], int)    # 반드시 integer로 선언
  a[b]
  ```
  `array([2, 2, 4, 8, 6, 4])`
  ```python
  a.take(b)
  ```
  `array([2, 2, 4, 8, 6, 4])`
  - Matrix 형태의 데이터도 가능 (많이 쓰지 않음)
  ```python
  a = np.array([[1,4],[9, 16]])
  b = np.array([0,0,1,1,0], int)
  c = np.array([0,1,1,0,1], int)
  a[b,c]
  ```
  `array([ 1,  4, 16,  9,  4])`
  
  [실행 코드 보기](./boolean&fancy_index.ipynb)
  
  
 ## numpy data i/o
  ### loadtxt & savetxt
  - Text type 의 데이터를 읽고, 저장하는 기능
  ```python
  a = np.loadtxt("./populations.txt")
  a[:10]
  ```
  ```
  array([[ 1900., 30000.,  4000., 48300.],
       [ 1901., 47200.,  6100., 48200.],
       [ 1902., 70200.,  9800., 41500.],
       [ 1903., 77400., 35200., 38200.],
       [ 1904., 36300., 59400., 40600.],
       [ 1905., 20600., 41700., 39800.],
       [ 1906., 18100., 19000., 38600.],
       [ 1907., 21400., 13000., 42300.],
       [ 1908., 22000.,  8300., 44500.],
       [ 1909., 25400.,  9100., 42100.]])
  ```
  ```python
  a_int = a.astype(int)
  a_int[:3]
  ```
  ```
  array([[ 1900, 30000,  4000, 48300],
       [ 1901, 47200,  6100, 48200],
       [ 1902, 70200,  9800, 41500]])
  ```
  ```python
  np.savetxt('int_data.csv',a_int, fmt='%d', delimiter=",")
  ```
  ### numpy object - npy
  - numpy object (pickle) 형태로 데이터를 저장하고 불러옴
  - Binary 파일 형태로 저장함
  ```python
  np.save("npy_test.npy", arr=a_int)
  ```
  ```python
  npy_array = np.load(file="npy_test.npy")
  npy_array[:3]
  ```
  ```
  array([[ 1900, 30000,  4000, 48300],
       [ 1901, 47200,  6100, 48200],
       [ 1902, 70200,  9800, 41500]])
  ```
  
  [실행 코드 보기](./numpy_data_io.ipynb)
