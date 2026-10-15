/**
 * engineRunner.js
 * ---------------
 * Python AI 추론 엔진(engine/main.py)을 서브프로세스로 실행하고
 * 표준 출력된 JSON 판정 결과를 파싱하여 반환하는 서비스
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

class EngineRunner {
  constructor() {
    this.rootDir = path.resolve(__dirname, '../../../');
    this.engineDir = path.join(this.rootDir, 'engine');
    this.engineScript = path.join(this.engineDir, 'main.py');

    // Windows venv 우선 탐지
    const venvPythonWin = path.join(this.rootDir, 'venv', 'Scripts', 'python.exe');
    const venvPythonUnix = path.join(this.rootDir, 'venv', 'bin', 'python');

    if (fs.existsSync(venvPythonWin)) {
      this.pythonPath = venvPythonWin;
    } else if (fs.existsSync(venvPythonUnix)) {
      this.pythonPath = venvPythonUnix;
    } else {
      this.pythonPath = 'python'; // 시스템 기본 파이썬
    }

    console.log(`[EngineRunner] Python 실행기 설정: ${this.pythonPath}`);
  }

  /**
   * 단일 비디오 클립을 Python 엔진에 전달하여 분석 실행
   * @param {string} videoAbsolutePath 비디오 파일의 절대 경로
   * @param {string} [roiCoords] "x,y,w,h" 형태의 수동 ROI 좌표 (선택)
   * @returns {Promise<Object>} 분석 결과 JSON 객체
   */
  runAnalysis(videoAbsolutePath, roiCoords = null) {
    return new Promise((resolve, reject) => {
      if (!fs.existsSync(videoAbsolutePath)) {
        return reject(new Error(`비디오 파일을 찾을 수 없습니다: ${videoAbsolutePath}`));
      }

      const args = [this.engineScript, '--video', videoAbsolutePath, '--json'];
      if (roiCoords) {
        args.push('--roi', roiCoords);
      }

      console.log(`[EngineRunner] 분석 작업 시작: ${this.pythonPath} ${args.join(' ')}`);

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
          // stdout 중 JSON 라인 파싱 ({...} 형태의 마지막 JSON 라인 탐색)
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

          // 스냅샷 파일명을 웹 서빙 가능한 URL 경로로 가공 (/outputs/filename.jpg)
          if (parsedResult.snapshot_path) {
            const snapshotFilename = path.basename(parsedResult.snapshot_path);
            parsedResult.snapshot_url = `/outputs/${snapshotFilename}`;
          } else {
            parsedResult.snapshot_url = null;
          }

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
