/**
 * server.js
 * ---------
 * Node.js 서버 실행 진입점
 */

require('dotenv').config();
const app = require('./app');

const PORT = process.env.PORT || 4000;

app.listen(PORT, () => {
  console.log('='.repeat(55));
  console.log(`[AI Hawk-Eye Lite] 백엔드 API 게이트웨이 구동 완료`);
  console.log(`- 서버 주소: http://localhost:${PORT}`);
  console.log(`- 헬스체크: http://localhost:${PORT}/api/health`);
  console.log(`- 판정 업로드: POST http://localhost:${PORT}/api/judge/upload`);
  console.log(`- 샘플 테스트: POST http://localhost:${PORT}/api/judge/test-sample`);
  console.log(`- 판정 이력: GET http://localhost:${PORT}/api/judge/history`);
  console.log('='.repeat(55));
});
