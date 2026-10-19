/**
 * historyStore.js
 * ---------------
 * 판정 이력 및 사용자 피드백을 메모리 캐시 + 원자적(Atomic) JSON 파일 저장으로 관리하고,
 * 오래된 임시 업로드/스냅샷 파일을 자동 정리(Retention Cleanup)하는 저장소
 */

const fs = require('fs');
const path = require('path');

const DATA_DIR = path.resolve(__dirname, '../../data');
const DATA_FILE = path.join(DATA_DIR, 'history.json');
const MAX_RETAINED_FILES = 100;

class HistoryStore {
  constructor() {
    this.cache = null;
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
    if (this.cache !== null) {
      return this.cache;
    }
    try {
      this._ensureFile();
      const content = fs.readFileSync(DATA_FILE, 'utf-8');
      this.cache = JSON.parse(content) || [];
      return this.cache;
    } catch (err) {
      console.error('[HistoryStore] 파일 읽기 실패:', err);
      this.cache = [];
      return this.cache;
    }
  }

  _writeAll(data) {
    this.cache = data;
    try {
      this._ensureFile();
      const tmpFile = `${DATA_FILE}.tmp`;
      fs.writeFileSync(tmpFile, JSON.stringify(data, null, 2), 'utf-8');
      fs.renameSync(tmpFile, DATA_FILE);
    } catch (err) {
      console.error('[HistoryStore] 파일 쓰기 실패:', err);
    }
  }

  saveRecord(record) {
    const list = this._readAll();
    const entry = {
      id: record.id,
      created_at: new Date().toISOString(),
      status: record.status || 'SUCCESS',
      video_filename: record.video_filename,
      physical_decision: record.physical_decision || 'UNCERTAIN',
      final_decision: record.final_decision || 'UNCERTAIN',
      match_status: record.match_status || 'CONFIRMED',
      distance_cm: record.distance_cm ?? 0.0,
      confidence: record.confidence ?? 0.0,
      impact_frame_idx: record.impact_frame_idx ?? 0,
      timestamp_ms: record.timestamp_ms ?? 0.0,
      contact_point: record.contact_point || [0, 0],
      snapshot_url: record.snapshot_url || null,
      commentary: record.commentary || null,
      feedback: null,
    };

    list.unshift(entry);
    this._writeAll(list);

    // 비동기로 오래된 임시 파일 정리
    setImmediate(() => this._cleanupOldMediaFiles());

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
      agreed: Boolean(feedbackData.agreed),
      user_decision: feedbackData.user_decision || null,
      comment: String(feedbackData.comment || '').slice(0, 500),
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

  /**
   * 디스크 용량 보호를 위해 보관 한도(MAX_RETAINED_FILES)를 초과한 오래된 업로드/스냅샷 정리
   */
  _cleanupOldMediaFiles() {
    try {
      const rootDir = path.resolve(__dirname, '../../../');
      const dirsToClean = [
        path.join(rootDir, 'server', 'uploads'),
        path.join(rootDir, 'engine', 'outputs'),
      ];

      for (const dir of dirsToClean) {
        if (!fs.existsSync(dir)) continue;
        const files = fs
          .readdirSync(dir)
          .filter((f) => f !== '.gitkeep')
          .map((name) => {
            const fullPath = path.join(dir, name);
            const stat = fs.statSync(fullPath);
            return { fullPath, mtimeMs: stat.mtimeMs };
          })
          .sort((a, b) => b.mtimeMs - a.mtimeMs);

        if (files.length > MAX_RETAINED_FILES) {
          for (const oldFile of files.slice(MAX_RETAINED_FILES)) {
            fs.unlinkSync(oldFile.fullPath);
          }
        }
      }
    } catch (err) {
      // 파일 정리 중 경미한 오류는 무시
    }
  }

  getDisputedCases() {
    const list = this._readAll();
    return list.filter((item) => item.feedback && item.feedback.agreed === false);
  }
}

module.exports = new HistoryStore();
