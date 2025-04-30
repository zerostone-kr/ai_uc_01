@echo off
chcp 65001 >nul
REM ✅ Git 자동 커밋 및 푸시 스크립트 (Windows CMD, UTF-8 인코딩)

REM 1. 현재 디렉토리 출력
cd

REM 2. 변경사항 확인
ECHO [1] Git 변경사항 확인...
git status

REM 3. 전체 파일 add
ECHO [2] 변경된 파일 스테이징...
git add .

REM 4. 커밋 메시지 입력 받기
SET /P msg=커밋 메시지를 입력하세요: 
IF "%msg%"=="" GOTO EmptyMessage

ECHO [3] 커밋 중...
git commit -m "%msg%"

REM 5. 푸시 실행
ECHO [4] GitHub로 푸시 중...
git push origin main

ECHO ✅ 작업 완료!
PAUSE
GOTO End

:EmptyMessage
ECHO ⚠️ 커밋 메시지가 비어 있어 작업을 취소합니다.
PAUSE

:End
