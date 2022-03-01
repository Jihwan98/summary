# Gihub 사용법
[Notion에 정리한 Gihub 사용법](https://www.notion.so/Github-a092c43efa6343dd8e9f4f1f88288a02)

## Git 사전 준비
`$ git config --global user.name '{이름}'`  
`$ git config --global user.email '{이메일}'`

## Git 기초 흐름
```
$ git init //저장소 설정
$ git add .
$ git commit -m '{커밋메시지}'
$ git branch -M main //default branch 가 main으로 변경되어서..
$ git remote add origin {url} //원격 저장소 설정 (원격저장소(remote)로 origin 이름으로 url을 추가))
$ git push -u origin main
```

## 각종 명령어
- git 상태 확인  
`$ git status`  
- branch 확인  
`$ git branch`  
- 원격 저장소 목록  
`$ git remote -v`  
- 원격 저장소 삭제  
`$ git remote rm origin`  

## 각종 에러 상황
- fatal: refusing to merge unrelated histories [관련 블로그 링크](https://gdtbgl93.tistory.com/63)  
`git pull origin 브랜치명 --allow-unrelated-histories`  
`--allow-unrelated-histories` 이 명령 옵션은 이미 존재하는 두 프로젝트의 기록(history)을 저장하는 드문 상황에 사용된다고 한다. 즉, git에서는 서로 관련 기록이 없는 이질적인 두 프로젝트를 병합할 때 기본적으로 거부하는데, 이를 허용해 주는 것이다.
