@echo off
echo ========================================================
echo [AI Hawk-Eye Lite] 스마트폰 외부 모바일 접속 터널 시작
echo ========================================================
echo.
echo 로컬 서버 포트: 4000
echo Cloudflare 무료 터널(npx cloudflared)을 실행합니다...
echo 잠시 후 터미널에 생성되는 https://xxxx.trycloudflare.com 주소를
echo 스마트폰 브라우저에 입력하여 코트 현장에서 접속하세요.
echo.
npx untun@latest tunnel --port 4000
pause
