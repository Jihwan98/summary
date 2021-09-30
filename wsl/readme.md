# WSL to Window

## 윈도우에서 Ubuntu 사용하기
1. Windows 기능 켜기/끄기 에서 `Linux용 Windows 하위 시스템` 체크 ! 이후 리부팅
2. Microsoft Store에서 Ubuntu 다운로드 (Ubuntu 20.04 LTS 등 상관없음)
3. 다운받은 Ubuntu 실행
4. 아이디와 비밀번호 생성해주면 끝.
5. `$ sudo apt-get update`, `$ sudo apt-get upgrade`
6. cmd나 anaconda prompt, miniconda prompt 등에서 bash 입력시 ubuntu 접속

## WSL Ubuntu에서 Miniconda 사용하기
1. [miniconda 다운 링크](ttps://docs.conda.io/en/latest/miniconda.html) 에서 다운링크 가져오기 (제일 최신것으로 하니까 에러나서 python3.8 버전으로 다운했음)
2. wget [download link]
3. 다운받은 파일 bash로 실행하기 `$ bash Mini~~`
4. ubuntu 껐다 켜면 conda 활성화됨.

이후 가상환경을 만들고 pip 실행시 에러가 뜨는데, proxy 연결에 의한 에러?(warning?)로 보인다
환경을 생성후 컴퓨터 자체를 껐다 켜면 문제없이 실행되고, sudo 권한으로 pip 진행하면 상관없이 실행된다 (ex: sudo pip install numpy) 

## Xming을 이용해서 GUI X Window 설정하기
1. [다운링크](https://sourceforge.net/projects/xming/) 에서 Xming 서버를 다운로드 받아 Windows에 설치한다.  
  (시작 프로그램 폴더 (시작 -> 실행 -> "shell:startup")에 Xming 단축 아이콘을 위치시켜 Windows 부팅시 자동으로 실행되도록 한다.)
2. Machine ID 생성  
  `$ sudo systemd-machine-id-setup`  
  `$ sudo dbus-uuidgen --ensure`  
  다음 명령으로 GUID가 올바르게 생성되었는지 확인한다.  
  `$ cat /etc/machine-id`  
3. X-Window 구성 요소 설치  
  `$ sudo apt-get install x11-apps xfonts-base xfonts-100dpi xfonts-75dpi xfonts-cyrillic`  
4. 기본 디스플레이 포트 설정  
  ~/.bashrc 에 디스플레이 환경 변수를 다음과 같이 설정한다.  
  `$ vi ~/.bashrc`  
  `export DISPLAY=:0`  
  WSL Shell을 종료하고 다시 실행하거나, `$ source ~/.bashrc` 명령을 실행하여 변경된 환경 변수를 적용한다.  
5. 디스플레이 동작 확인  
  `$ xeyes`  
  ![image](https://user-images.githubusercontent.com/76936390/134667389-c607fa3d-1187-44f1-a980-1de845d646dd.png)
  
| wsl2로 변경 후 위와 같은 방법으로 했을 경우 X window가 설정되지 않았다. 따라서 아래와 같이 해결하였다. [참고링크](https://evandde.github.io/wsl2-x/)
1. Xming 설치는 동일  
  Xming 단축 아이콘에서 속성-바로가기-대상 항목에 맨 끝에 한칸을 띄우고 -ac 를 입력  
  ![image](https://user-images.githubusercontent.com/76936390/135489811-7d935950-6a4e-4b0a-a164-9ad44cc03e1b.png)
  Xming이 켜져있다면 종료하고 수정한 바로가기로 Xming을 실행해준다.
2. windows powershell을 관리자 권한으로 실행 후 `Set-NetFirewallRule -DisplayName "Xming X Server" -Enabled True -Profile Any` 입력  
  Xming을 실행하지 않고 입력하면 에러가 뜨는데, 실행하고 나서도 에러가 뜬다면 `New-NetFirewallRule -DisplayName "Xming X Server" -Enabled True -Profile Any` 을 입력.
3. wsl에서 ~/.bashrc 에 디스플레이 환경 변수를 다음과 같이 설정한다.  
  `export DISPLAY=$(cat /etc/resolv.conf |grep nameserver | awk '{print $2}'):0`
  WSL Shell을 종료하고 다시 실행하거나, `$ source ~/.bashrc` 명령을 실행하여 변경된 환경 변수를 적용한다.
4. `xeyes` 나 `xclock`을 통해 디스플레이 동작을 확인한다.
  
## jupyter notebook
wsl ubuntu 환경에서 jupyter notebook 실행 시, browser가 안뜨고 link 도 순식간에 넘어가는데 이를 해결 하는 방법
![캡처](https://user-images.githubusercontent.com/76936390/133366287-d0114418-ac5c-45cc-994a-618961578d49.PNG)

## VS Code
Remote-WSL 다운로드
