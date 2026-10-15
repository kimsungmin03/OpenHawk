/**
 * historyStore.js
 * ---------------
 * 판정 이력 및 사용자 피드백을 JSON 파일 기반으로 영속 저장/조회하는 경량 저장소
 */

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.resolve(__dirname, '../../data');
const DATA_FILE = path.join(DATA_DIR, 'history.json');

class HistoryStore {
  constructor() {
    this._ensureFile();
  }

  _ensureFile() {
    if (!fs.existsSync(DATA_DIR)) {
      fs.mkdirSync(DATA_DIR, { recursive: true });
    }
    if (!fs.existsSync(DATA_FILE)) {
      fs.writeFileSync(DATA_FILE, JSON.stringify([], null, 2), 'utf-8');
    }
  }

  _readAll() {
    try {
      this._ensureFile();
      const content = fs.readFileSync(DATA_FILE, 'utf-8');
      return JSON.parse(content) || [];
    } catch (err) {
      console.error('[HistoryStore] 파일 읽기 실패:', err);
      return [];
    }
  }

  _writeAll(data) {
    try {
      fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2), 'utf-8');
    } catch (err) {
      console.error('[HistoryStore] 파일 쓰기 실패:', err);
    }
  }

  saveRecord(record) {
    const list = this._readAll();
    const entry = {
      id: record.id,
      created_at: new Date().toISOString(),
      video_filename: record.video_filename,
      physical_decision: record.physical_decision,
      final_decision: record.final_decision,
      match_status: record.match_status,
      distance_cm: record.distance_cm,
      confidence: record.confidence,
      impact_frame_idx: record.impact_frame_idx,
      contact_point: record.contact_point,
      snapshot_url: record.snapshot_url,
      commentary: record.commentary,
      feedback: null, // 초기 피드백은 null
    };

    list.unshift(entry); // 최신순
    this._writeAll(list);
    return entry;
  }

  getRecord(id) {
    const list = this._readAll();
    return list.find((item) => item.id === id) || null;
  }

  getAllRecords(limit = 50) {
    const list = this._readAll();
    return list.slice(0, limit);
  }

  updateFeedback(id, feedbackData) {
    const list = this._readAll();
    const index = list.findIndex((item) => item.id === id);
    if (index === -1) {
      return null;
    }

    const feedbackObj = {
      agreed: Boolean(feedbackData.agreed), // true: 판정인정, false: 판정불복
      user_decision: feedbackData.user_decision || null, // 사용자가 생각한 실제 결과 (IN / OUT)
      comment: feedbackData.comment || '',
      submitted_at: new Date().toISOString(),
    };

    list[index].feedback = feedbackObj;
    this._writeAll(list);

    // [Active Learning] 판정 불복 건(오심 의심 케이스) 자동 아카이빙
    if (!feedbackObj.agreed) {
      this._archiveHardNegative(list[index]);
    }

    return list[index];
  }

  /**
   * 오심 의심 케이스를 dataset/hard-negatives 폴더에 자동 격리 저장
   */
  _archiveHardNegative(record) {
    try {
      const rootDir = path.resolve(__dirname, '../../../');
      const targetDir = path.join(rootDir, 'dataset', 'hard-negatives', record.id);
      fs.mkdirSync(targetDir, { recursive: true });

      // 1. 스냅샷 복사
      if (record.snapshot_url) {
        const snapshotFilename = path.basename(record.snapshot_url);
        const sourceSnapshot = path.join(rootDir, 'engine', 'outputs', snapshotFilename);
        if (fs.existsSync(sourceSnapshot)) {
          fs.copyFileSync(sourceSnapshot, path.join(targetDir, snapshotFilename));
        }
      }

      // 2. 비디오 클립 복사
      if (record.video_filename) {
        // uploads 폴더 또는 assets/videos 폴더 탐색
        const uploadVideo = path.join(rootDir, 'server', 'uploads', record.video_filename);
        const assetVideo = path.join(rootDir, 'engine', 'assets', 'videos', record.video_filename);

        if (fs.existsSync(uploadVideo)) {
          fs.copyFileSync(uploadVideo, path.join(targetDir, record.video_filename));
        } else if (fs.existsSync(assetVideo)) {
          fs.copyFileSync(assetVideo, path.join(targetDir, record.video_filename));
        }
      }

      // 3. 메타데이터 및 교정 라벨 JSON 저장
      const metadataPath = path.join(targetDir, 'case_info.json');
      fs.writeFileSync(metadataPath, JSON.stringify(record, null, 2), 'utf-8');

      console.log(`[ActiveLearning] 오심 케이스 격리 저장 완료: dataset/hard-negatives/${record.id}`);
    } catch (err) {
      console.error('[ActiveLearning] 오심 아카이빙 실패:', err);
    }
  }

  getDisputedCases() {
    const list = this._readAll();
    return list.filter((item) => item.feedback && item.feedback.agreed === false);
  }
}

module.exports = new HistoryStore();
