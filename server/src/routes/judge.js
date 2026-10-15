/**
 * judge.js
 * --------
 * 비디오 클립 업로드 및 판독 실행, 판정 이력 조회 API 라우터
 */

const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

const engineRunner = require('../services/engineRunner');
const historyStore = require('../services/historyStore');

// 비디오 파일 업로드 디렉토리 설정
const uploadDir = path.resolve(__dirname, '../../uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// Multer 스토리지 설정
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname) || '.mp4';
    const uniqueName = `clip_${Date.now()}_${uuidv4().slice(0, 8)}${ext}`;
    cb(null, uniqueName);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 100 * 1024 * 1024 }, // 최대 100MB
  fileFilter: (req, file, cb) => {
    const allowed = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowed.includes(ext) || file.mimetype.startsWith('video/')) {
      cb(null, true);
    } else {
      cb(new Error('동영상 파일(.mp4, .mov, .avi 등)만 업로드 가능합니다.'));
    }
  },
});

/**
 * POST /api/judge/upload
 * 비디오 클립 업로드 후 AI 판정 즉시 실행
 */
router.post('/upload', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: '업로드된 비디오 파일이 없습니다.' });
    }

    const videoPath = req.file.path;
    const roiCoords = req.body.roi || null; // 예: "100,200,400,100"

    console.log(`[Judge API] 새 비디오 수신: ${req.file.filename} (${req.file.size} bytes)`);

    // 1. Python 추론 엔진 실행
    const engineResult = await engineRunner.runAnalysis(videoPath, roiCoords);

    // 2. 이력 저장
    const recordId = uuidv4();
    const record = historyStore.saveRecord({
      id: recordId,
      video_filename: req.file.filename,
      physical_decision: engineResult.physical_decision,
      final_decision: engineResult.final_decision,
      match_status: engineResult.match_status,
      distance_cm: engineResult.distance_cm,
      confidence: engineResult.confidence,
      impact_frame_idx: engineResult.impact_frame_idx,
      contact_point: engineResult.contact_point,
      snapshot_url: engineResult.snapshot_url,
      commentary: engineResult.commentary,
    });

    return res.status(200).json({
      success: true,
      data: record,
      raw_engine_result: engineResult,
    });
  } catch (err) {
    console.error('[Judge API] 판정 처리 오류:', err);
    return res.status(500).json({
      success: false,
      error: err.message || '영상 분석 처리 중 오류가 발생했습니다.',
    });
  }
});

/**
 * POST /api/judge/test-sample
 * 서버에 준비된 가상 샘플 클립(sample_in.mp4 / sample_out.mp4)으로 즉시 테스트
 */
router.post('/test-sample', async (req, res) => {
  try {
    const sampleType = req.body.sampleType === 'out' ? 'sample_out.mp4' : 'sample_in.mp4';
    const samplePath = path.resolve(__dirname, '../../../engine/assets/videos', sampleType);

    if (!fs.existsSync(samplePath)) {
      return res.status(404).json({ error: `샘플 비디오를 찾을 수 없습니다: ${sampleType}` });
    }

    console.log(`[Judge API] 샘플 테스트 요청 실행: ${sampleType}`);
    const engineResult = await engineRunner.runAnalysis(samplePath);

    const recordId = uuidv4();
    const record = historyStore.saveRecord({
      id: recordId,
      video_filename: sampleType,
      physical_decision: engineResult.physical_decision,
      final_decision: engineResult.final_decision,
      match_status: engineResult.match_status,
      distance_cm: engineResult.distance_cm,
      confidence: engineResult.confidence,
      impact_frame_idx: engineResult.impact_frame_idx,
      contact_point: engineResult.contact_point,
      snapshot_url: engineResult.snapshot_url,
      commentary: engineResult.commentary,
    });

    return res.status(200).json({
      success: true,
      data: record,
    });
  } catch (err) {
    console.error('[Judge API] 샘플 테스트 오류:', err);
    return res.status(500).json({ success: false, error: err.message });
  }
});

/**
 * GET /api/judge/history
 * 판정 이력 전체 목록 조회
 */
router.get('/history', (req, res) => {
  try {
    const limit = parseInt(req.query.limit, 10) || 50;
    const history = historyStore.getAllRecords(limit);
    return res.status(200).json({ success: true, count: history.length, data: history });
  } catch (err) {
    return res.status(500).json({ success: false, error: err.message });
  }
});

/**
 * GET /api/judge/:id
 * 특정 판정 레코드 상세 조회
 */
router.get('/:id', (req, res) => {
  try {
    const record = historyStore.getRecord(req.params.id);
    if (!record) {
      return res.status(404).json({ success: false, error: '판정 기록을 찾을 수 없습니다.' });
    }
    return res.status(200).json({ success: true, data: record });
  } catch (err) {
    return res.status(500).json({ success: false, error: err.message });
  }
});

module.exports = router;
