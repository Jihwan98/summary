# WSL to Window

## 윈도우에서 Ubuntu 사용하기
1. Windows 기능 켜기/끄기 에서 `Linux용 Windows 하위 시스템` 체크 ! 이후 리부팅
2. Microsoft Store에서 Ubuntu 다운로드 (Ubuntu 20.04 LTS 등 상관없음)
3. 다운받은 Ubuntu 실행
4. 아이디와 비밀번호 생성해주면 끝.
5. `$ sudo apt-get update`, `$ sudo apt-get upgrade`
6. cmd나 anaconda prompt, miniconda prompt 등에서 bash 입력시 ubuntu 접속

## WSL Ubuntu에서 Miniconda 사용하기
1. https://docs.conda.io/en/latest/miniconda.html 에서 다운링크 가져오기 (제일 최신것으로 하니까 에러나서 python3.8 버전으로 다운했음)
2. wget [download link]
3. 다운받은 파일 bash로 실행하기 `$ bash Mini~~`
4. ubuntu 껐다 켜면 conda 활성화됨.

이후 가상환경을 만들고 pip 실행시 에러가 뜨는데, proxy 연결에 의한 에러?(warning?)로 보인다
환경을 생성후 컴퓨터 자체를 껐다 켜면 문제없이 실행되고, sudo 권한으로 pip 진행하면 상관없이 실행된다 (ex: sudo pip install numpy) 

## jupyter notebook
wsl ubuntu 환경에서 jupyter notebook 실행 시, browser가 안뜨고 link 도 순식간에 넘어가는데 이를 해결 하는 방법
![캡처](https://user-images.githubusercontent.com/76936390/133366287-d0114418-ac5c-45cc-994a-618961578d49.PNG)

## VS Code
Remote-WSL 다운로드
