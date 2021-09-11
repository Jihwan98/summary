# Anaconda 및 Miniconda 사용법
## Conda 버전 확인
- `conda -V`
- `conda --version`

## 가상환경 생성
```
$ conda create -n test_env python=3.5
```
Python 3.5 버전의 'test_env'라는 이름으로 `env`를 생성

## env list 보기
```
$ conda env list
```

## env 활성화
```
$ conda activate test_env
```

## env 비활성화
```
$ conda deactivate
```

## env 삭제
```
$ conda env remove -n test_env
```

# jupyter notebook
## jupyter notebook install
```
$ pip insatll jupyter
```

## ipykernel (가상환경 별로 jupyter kernel에 연결시켜줘야함)
```
$ pip install ipykernel
```
```
$ python -m ipykernel install --user --name=가상환경이름
```

## 제거하기
```
$ jupyter kernelspec uninstall 가상환경이름
```
