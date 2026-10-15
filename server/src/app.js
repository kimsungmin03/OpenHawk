/**
 * app.js
 * ------
 * Express 애플리케이션 미들웨어, 정적 파일 서빙 및 라우트 등록
 */

const express = require('express');
const cors = require('cors');
const path = require('path');

const judgeRouter = require('./routes/judge');
const feedbackRouter = require('./routes/feedback');

const app = express();

// 미들웨어
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// 호크아이 판독 스냅샷 이미지 정적 파일 서빙 (/outputs/hawkeye_in_xxx.jpg)
const engineOutputsDir = path.resolve(__dirname, '../../engine/outputs');
app.use('/outputs', express.static(engineOutputsDir));

// 업로드된 비디오 파일 정적 파일 서빙 (/uploads/clip_xxx.mp4)
const serverUploadsDir = path.resolve(__dirname, '../uploads');
app.use('/uploads', express.static(serverUploadsDir));

// 프론트엔드 모바일 웹 클라이언트 정적 서빙
const clientDir = path.resolve(__dirname, '../../client');
app.use(express.static(clientDir));

// API 라우터
app.use('/api/judge', judgeRouter);
app.use('/api/feedback', feedbackRouter);

// 헬스체크
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ONLINE',
    service: 'AI Hawk-Eye Lite API Gateway',
    timestamp: new Date().toISOString(),
  });
});

module.exports = app;
