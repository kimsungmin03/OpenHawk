/**
 * engineRunner.js
 * ---------------
 * Python AI 추론 엔진을 상주형 워커(engine/worker.py)로 유지하여
 * YOLOv8/CUDA Cold-Start 지연 없이 초고속으로 비디오 판독을 수행하는 서비스.
 * 워커 장애 시 단일 프로세스(engine/main.py) 모드로 자동 폴백합니다.
 */

const { spawn } = require('child_process');
const readline = require('readline');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

class EngineRunner {
  constructor() {
    this.rootDir = path.resolve(__dirname, '../../../');
    this.engineDir = path.join(this.rootDir, 'engine');
    this.engineScript = path.join(this.engineDir, 'main.py');
    this.workerScript = path.join(this.engineDir, 'worker.py');

    // Windows venv 우선 탐지
    const venvPythonWin = path.join(this.rootDir, 'venv', 'Scripts', 'python.exe');
    const venvPythonUnix = path.join(this.rootDir, 'venv', 'bin', 'python');

    if (fs.existsSync(venvPythonWin)) {
      this.pythonPath = venvPythonWin;
    } else if (fs.existsSync(venvPythonUnix)) {
      this.pythonPath = venvPythonUnix;
    } else {
      this.pythonPath = 'python';
    }

    this.workerProcess = null;
    this.workerReady = false;
    this.workerReadyPromise = null;
    this.pendingRequests = new Map();

    console.log(`[EngineRunner] Python 실행기 설정: ${this.pythonPath}`);
    this._startPersistentWorker();
  }

  /**
   * 상주형 Python 워커 프로세스 시작 및 JSON-RPC 스트림 연결
   */
  _startPersistentWorker() {
    if (!fs.existsSync(this.workerScript)) {
      return;
    }

    let resolveReady;
    this.workerReadyPromise = new Promise((resolve) => {
      resolveReady = resolve;
    });

    try {
      this.workerReady = false;
      const child = spawn(this.pythonPath, [this.workerScript], {
        cwd: this.rootDir,
        env: {
          ...process.env,
          PYTHONIOENCODING: 'utf-8',
        },
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      this.workerProcess = child;

      const rl = readline.createInterface({
        input: child.stdout,
        crlfDelay: Infinity,
      });

      rl.on('line', (line) => {
        const trimmed = line.trim();
        if (!trimmed.startsWith('{')) return;

        try {
          const msg = JSON.parse(trimmed);
          if (msg.type === 'READY') {
            this.workerReady = true;
            resolveReady(true);
            console.log(`[EngineRunner] 상주형 AI 워커 준비 완료 (backend=${msg.backend}, device=${msg.device})`);
            return;
          }

          if (msg.id && this.pendingRequests.has(msg.id)) {
            const { resolve, reject, timer } = this.pendingRequests.get(msg.id);
            clearTimeout(timer);
            this.pendingRequests.delete(msg.id);

            if (msg.ok) {
              resolve(this._formatSnapshotUrl(msg.result));
            } else {
              reject(new Error(msg.error || '추론 워커 처리 실패'));
            }
          }
        } catch (err) {
          console.error('[EngineRunner] 워커 응답 파싱 오류:', err);
        }
      });

      child.stderr.on('data', (data) => {
        const logText = data.toString('utf-8').trim();
        if (logText) {
          console.log(logText);
        }
      });

      child.on('close', (code) => {
        this.workerReady = false;
        this.workerProcess = null;
        resolveReady(false);
        console.warn(`[EngineRunner] 상주형 워커 종료 (code=${code}). 대기 중인 요청 폴백 처리`);

        for (const [, pending] of this.pendingRequests.entries()) {
          clearTimeout(pending.timer);
          pending.reject(new Error('Python 워커가 비정상 종료되었습니다.'));
        }
        this.pendingRequests.clear();
      });

      child.on('error', (err) => {
        console.error('[EngineRunner] 상주형 워커 실행 에러:', err.message);
        this.workerReady = false;
        this.workerProcess = null;
        resolveReady(false);
      });
    } catch (err) {
      console.error('[EngineRunner] 상주형 워커 초기화 실패:', err.message);
      resolveReady(false);
    }
  }

  _formatSnapshotUrl(parsedResult) {
    if (parsedResult && parsedResult.snapshot_path) {
      const snapshotFilename = path.basename(parsedResult.snapshot_path);
      parsedResult.snapshot_url = `/outputs/${snapshotFilename}`;
    } else if (parsedResult) {
      parsedResult.snapshot_url = null;
    }
    return parsedResult;
  }

  /**
   * 단일 비디오 클립을 Python 엔진에 전달하여 분석 실행
   * @param {string} videoAbsolutePath 비디오 파일의 절대 경로
   * @param {string} [roiCoords] "x,y,w,h" 형태의 수동 ROI 좌표 (선택)
   * @returns {Promise<Object>} 분석 결과 JSON 객체
   */
  async runAnalysis(videoAbsolutePath, roiCoords = null) {
    if (!fs.existsSync(videoAbsolutePath)) {
      throw new Error(`비디오 파일을 찾을 수 없습니다: ${videoAbsolutePath}`);
    }

    if (!this.workerProcess) {
      this._startPersistentWorker();
    }

    if (this.workerProcess && !this.workerReady && this.workerReadyPromise) {
      await this.workerReadyPromise;
    }

    // 상주형 워커가 활성 상태이면 즉시 파이프로 전달 (Cold-Start 없음)
    if (this.workerProcess && this.workerReady && this.workerProcess.stdin.writable) {
      try {
        return await this._runViaWorker(videoAbsolutePath, roiCoords);
      } catch (err) {
        console.warn(`[EngineRunner] 워커 요청 실패(${err.message}), 일회성 프로세스로 재시도합니다.`);
        this._startPersistentWorker();
      }
    }

    return this._runOneShotProcess(videoAbsolutePath, roiCoords);
  }

  _runViaWorker(videoAbsolutePath, roiCoords = null) {
    return new Promise((resolve, reject) => {
      const reqId = uuidv4();
      const timer = setTimeout(() => {
        if (this.pendingRequests.has(reqId)) {
          this.pendingRequests.delete(reqId);
          reject(new Error('상주형 워커 응답 타임아웃 (45초)'));
        }
      }, 45000);

      this.pendingRequests.set(reqId, { resolve, reject, timer });

      const payload = JSON.stringify({
        id: reqId,
        cmd: 'analyze',
        video_path: videoAbsolutePath,
        roi_coords: roiCoords,
      });

      this.workerProcess.stdin.write(payload + '\n', 'utf-8', (err) => {
        if (err) {
          clearTimeout(timer);
          this.pendingRequests.delete(reqId);
          reject(err);
        }
      });
    });
  }

  _runOneShotProcess(videoAbsolutePath, roiCoords = null) {
    return new Promise((resolve, reject) => {
      const args = [this.engineScript, '--video', videoAbsolutePath, '--json'];
      if (roiCoords) {
        args.push('--roi', roiCoords);
      }

      console.log(`[EngineRunner] 단일 프로세스 분석 시작: ${this.pythonPath} ${args.join(' ')}`);

      const child = spawn(this.pythonPath, args, {
        cwd: this.rootDir,
        env: {
          ...process.env,
          PYTHONIOENCODING: 'utf-8',
        },
      });

      let stdoutData = '';
      let stderrData = '';

      child.stdout.on('data', (data) => {
        stdoutData += data.toString('utf-8');
      });

      child.stderr.on('data', (data) => {
        stderrData += data.toString('utf-8');
      });

      child.on('close', (code) => {
        if (code !== 0) {
          console.error('[EngineRunner] Python 프로세스 오류 종료. Code:', code, stderrData);
          return reject(new Error(`추론 엔진 오류 (code: ${code}): ${stderrData || stdoutData}`));
        }

        try {
          const lines = stdoutData.trim().split('\n');
          let parsedResult = null;

          for (let i = lines.length - 1; i >= 0; i--) {
            const line = lines[i].trim();
            if (line.startsWith('{') && line.endsWith('}')) {
              try {
                parsedResult = JSON.parse(line);
                break;
              } catch (e) {
                // 계속 탐색
              }
            }
          }

          if (!parsedResult) {
            throw new Error('엔진 출력에서 유효한 JSON을 찾을 수 없습니다.\n출력: ' + stdoutData);
          }

          this._formatSnapshotUrl(parsedResult);
          console.log(`[EngineRunner] 분석 완료! 판정: ${parsedResult.final_decision} (${parsedResult.video_file})`);
          resolve(parsedResult);
        } catch (err) {
          console.error('[EngineRunner] JSON 파싱 실패:', err, stdoutData);
          reject(new Error(`결과 데이터 파싱 실패: ${err.message}`));
        }
      });

      child.on('error', (err) => {
        console.error('[EngineRunner] 프로세스 실행 에러:', err);
        reject(err);
      });
    });
  }
}

module.exports = new EngineRunner();
