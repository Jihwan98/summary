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
