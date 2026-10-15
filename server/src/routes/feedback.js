/**
 * feedback.js
 * -----------
 * 사용자 판정 피드백 수집 및 Active Learning 오심 격리 큐 API
 */

const express = require('express');
const router = express.Router();
const historyStore = require('../services/historyStore');

/**
 * POST /api/feedback
 * 판정 결과에 대한 사용자 피드백 제출
 * Body: { recordId, agreed, user_decision, comment }
 */
router.post('/', (req, res) => {
  try {
    const { recordId, agreed, user_decision, comment } = req.body;

    if (!recordId) {
      return res.status(400).json({ success: false, error: 'recordId가 필요합니다.' });
    }

    const updated = historyStore.updateFeedback(recordId, {
      agreed,
      user_decision,
      comment,
    });

    if (!updated) {
      return res.status(404).json({ success: false, error: '해당 판정 기록을 찾을 수 없습니다.' });
    }

    const statusMsg = agreed ? '판정 동의(인정)가 기록되었습니다.' : '오심 불복 피드백이 접수되었습니다. (Active Learning 리뷰 큐 등록)';
    console.log(`[Feedback API] 피드백 접수 - ID: ${recordId}, 동의여부: ${agreed}`);

    return res.status(200).json({
      success: true,
      message: statusMsg,
      data: updated,
    });
  } catch (err) {
    return res.status(500).json({ success: false, error: err.message });
  }
});

/**
 * GET /api/feedback/review-queue
 * 사용자가 불복한(오심 의심) 케이스 목록 조회 (Active Learning 데이터셋용)
 */
router.get('/review-queue', (req, res) => {
  try {
    const disputed = historyStore.getDisputedCases();
    return res.status(200).json({
      success: true,
      count: disputed.length,
      data: disputed,
    });
  } catch (err) {
    return res.status(500).json({ success: false, error: err.message });
  }
});

module.exports = router;
