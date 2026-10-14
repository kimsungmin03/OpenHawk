# 🏸 AI Hawk-Eye Lite (스마트 선심)
> **동호인 배드민턴 경기를 위한 엣지 기반 AI 베이스라인 판독 및 해설 시스템**  
> *AI-Powered Baseline Line-Call & Analysis System for Amateur Badminton*

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NodeJS](https://img.shields.io/badge/Node.js-18+-339933?logo=nodedotjs&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_Accelerated-EE4C2C?logo=pytorch&logoColor=white)

---

## 📌 1. 프로젝트 개요 (Overview)

배드민턴 경기 중 가장 빈번하게 발생하는 **베이스라인 인(In) / 아웃(Out) 판정 시비**를 해결하기 위한 보조 선심 시스템입니다.  
고가의 상용 호크아이 장비 대신, **스마트폰/액션캠 1대와 데스크톱 GPU 파이프라인**을 결합하여 합리적인 비용으로 정밀한 판정과 자연어 해설 리포트를 제공합니다.

### ✨ 주요 특징
* **포커스드 뷰(Focused View):** 코트 전체 대신 베이스라인 영역(ROI)에 집중하여 셔틀콕 크기 축소 및 모션 블러 문제 극복
* **고속 AI 추론 파이프라인:** OpenCV 기반 궤적 분석 + YOLOv8 GPU 가속을 통해 3~5초 분량의 랠리 클립을 1~2초 내에 판정
* **멀티모달 2차 검증 & 해설:** VLM(Vision-Language Model) 기반의 착지 스냅샷 정밀 판별 및 동호인 맞춤형 자연어 리포트 자동 생성
* **Active Learning 기반 개선:** 사용자 피드백(정확함/오심) 및 예외 로그를 수집하여 모델을 지속적으로 고도화하는 피드백 루프 지원

---

## 🏗️ 2. 시스템 아키텍처 (Architecture)

```text
[클라이언트: 모바일 UI]
  └─ 베이스라인 촬영 (60/120fps) & 3~5초 클립 전송
        │
        ▼  HTTPS (Cloudflare Tunnel / ngrok)
[백엔드 API 게이트웨이: Node.js (Express)]
  ├─ 멀티파트 비디오 업로드 처리 & 메타데이터 파싱
  ├─ 작업 스케줄링 및 결과 캐싱
  └─ 사용자 피드백 DB 저장
        │
        ▼  IPC / Local REST API
[AI 추론 엔진: Python 워커 (CUDA GPU 가속)]
  ├─ OpenCV: 베이스라인 관심 영역(ROI) 크롭 및 전처리
  ├─ 셔틀콕 추적 (YOLOv8 + 궤적 분석)
  ├─ 착지(Impact Frame) 검출 및 물리 좌표 연산
  └─ In/Out 판정 & 스냅샷 추출
        │
        ▼  API 요청
[생성형 AI: VLM & LLM]
  ├─ 착지 접촉점 교차 검증 (VLM)
  └─ 동호인용 친화적 판정 브리핑 멘트 생성 (LLM)
```

---

## 🛠️ 3. 기술 스택 (Tech Stack)

| 구분 | 기술 스택 | 설명 |
| :--- | :--- | :--- |
| **Frontend** | React Native / Mobile Web (Vite) | 모바일 클립 업로드, 판정 UI, 리플레이 뷰어 |
| **Backend** | Node.js (Express), SQLite / Prisma | API 게이트웨이, 작업 큐 관리, 이력 저장 |
| **Computer Vision** | OpenCV (`opencv-python`), NumPy | ROI 설정, 색상/이진화 필터, 영상 디코딩 |
| **Deep Learning** | PyTorch, Ultralytics YOLOv8 (CUDA) | 초고속 셔틀콕 Bounding Box 탐지 |
| **GenAI** | OpenAI GPT-4o / Google Gemini API | 착지 스냅샷 2차 검증 및 판정 해설 생성 |
| **Tunneling** | Cloudflare Tunnel / ngrok | 로컬 GPU 서버 외부 모바일 연동용 터널 |

---

## 📂 4. 디렉토리 구조 (Directory Structure)

```text
ai-hawkeye-lite/
├── client/                     # 모바일 프론트엔드 프로젝트
│   ├── src/
│   │   ├── components/        # 판정 카드, 카메라 뷰어 UI
│   │   └── api/               # 백엔드 통신 모듈
│   └── package.json
├── server/                     # Node.js 백엔드 게이트웨이
│   ├── src/
│   │   ├── routes/            # 업로드, 판정 조회, 피드백 API
│   │   └── controllers/
│   ├── uploads/               # 임시 저장 영상 클립
│   └── package.json
├── engine/                     # Python 비전/AI 추론 엔진
│   ├── models/                # YOLOv8 가중치 (.pt)
│   ├── core/
│   │   ├── tracker.py         # 셔틀콕 궤적 추적
│   │   ├── baseline_roi.py    # 베이스라인 매핑 & 착지 계산
│   │   └── commentary.py      # LLM 해설 생성 모듈
│   ├── main.py                # 엔진 실행 진입점
│   └── requirements.txt
└── README.md
```

---

## 🚀 5. 시작하기 (Quick Start)

### 1) 환경 요구사항
* **OS:** Windows 10/11 또는 Linux
* **Python:** 3.10 이상
* **Node.js:** 18.x 이상
* **GPU:** NVIDIA 외장 그래픽 (CUDA 가속 권장, 최소 VRAM 6GB 이상)

### 2) AI 추론 엔진 설정 (Python)
```bash
cd engine
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
python main.py
```

### 3) 백엔드 서버 설정 (Node.js)
```bash
cd ../server
npm install
npm run dev
```

### 4) 클라이언트 구동
```bash
cd ../client
npm install
npm run start
```

---

## 🔄 6. 피드백 루프 (Active Learning Loop)

1. 사용자가 모바일 UI에서 판정 결과에 대해 **[판정이 틀려요 👎]** 선택
2. 당시 영상 클립 및 계산된 셔틀콕 좌표가 백엔드 `review-queue`에 자동 격리
3. 오심 케이스 라벨링 보정 후 하드 네거티브(Hard Negative) 데이터셋에 추가
4. 주기적 파인튜닝(Fine-tuning)을 통해 지속적으로 판정 정확도 향상

---

## 📄 7. 라이선스 (License)

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.
