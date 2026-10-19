/**
 * app.js
 * ------
 * AI Hawk-Eye Lite 모바일 웹 클라이언트 스크립트
 */

document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }

  // DOM 요소 참조
  const videoInput = document.getElementById('video-input');
  const btnSampleIn = document.getElementById('btn-sample-in');
  const btnSampleOut = document.getElementById('btn-sample-out');
  const btnRefreshHistory = document.getElementById('btn-refresh-history');

  const loadingCard = document.getElementById('loading-card');
  const loadingStep = document.getElementById('loading-step');
  const resultCard = document.getElementById('result-card');

  const verdictBanner = document.getElementById('verdict-banner');
  const verdictText = document.getElementById('verdict-text');
  const matchStatusBadge = document.getElementById('match-status-badge');
  const marginText = document.getElementById('margin-text');
  const snapshotImg = document.getElementById('snapshot-img');
  const impactInfoText = document.getElementById('impact-info-text');

  const summaryText = document.getElementById('summary-text');
  const commentaryText = document.getElementById('commentary-text');
  const engineTypeTag = document.getElementById('engine-type-tag');

  const btnFeedbackAgree = document.getElementById('btn-feedback-agree');
  const btnFeedbackDisagree = document.getElementById('btn-feedback-disagree');
  const feedbackButtons = document.getElementById('feedback-buttons');
  const feedbackCompletedMsg = document.getElementById('feedback-completed-msg');

  const historyList = document.getElementById('history-list');
  const historyCount = document.getElementById('history-count');

  const imageModal = document.getElementById('image-modal');
  const modalImg = document.getElementById('modal-img');
  const btnCloseModal = document.getElementById('btn-close-modal');

  let currentRecordId = null;

  // 1. 영상 파일 업로드 처리
  videoInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('video', file);

    showLoading('영상을 업로드하고 AI 분석 파이프라인을 실행 중입니다...');

    try {
      const res = await fetch('/api/judge/upload', {
        method: 'POST',
        body: formData,
      });
      const json = await res.json();

      if (!res.ok || !json.success) {
        throw new Error(json.error || '영상 분석 실패');
      }

      renderResult(json.data);
      loadHistory();
    } catch (err) {
      alert(`판정 실패: ${err.message}`);
    } finally {
      hideLoading();
      videoInput.value = ''; // 재선택 가능하도록 리셋
    }
  });

  // 2. 가상 샘플 원클릭 시연 버튼
  btnSampleIn.addEventListener('click', () => runSampleTest('in'));
  btnSampleOut.addEventListener('click', () => runSampleTest('out'));

  async function runSampleTest(sampleType) {
    showLoading(`[${sampleType.toUpperCase()}] 가상 랠리 클립 시뮬레이션 중...`);
    try {
      const res = await fetch('/api/judge/test-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sampleType }),
      });
      const json = await res.json();
      if (!res.ok || !json.success) {
        throw new Error(json.error || '샘플 분석 실패');
      }
      renderResult(json.data);
      loadHistory();
    } catch (err) {
      alert(`샘플 테스트 오류: ${err.message}`);
    } finally {
      hideLoading();
    }
  }

  function escapeHtml(str) {
    return String(str ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // 3. 판정 결과 렌더링
  function renderResult(data) {
    currentRecordId = data.id;
    const decision = data.final_decision || 'UNCERTAIN';
    const isIN = decision === 'IN';
    const isUncertain = decision === 'UNCERTAIN' || data.status === 'FAIL';

    // 뱃지 스타일 업데이트
    if (isUncertain) {
      verdictBanner.className = 'p-5 flex items-center justify-between text-white bg-amber-600';
      verdictText.textContent = '판독 불가';
      marginText.textContent = '- cm';
    } else if (isIN) {
      verdictBanner.className = 'p-5 flex items-center justify-between text-white bg-emerald-600 neon-in';
      verdictText.textContent = 'IN';
      marginText.textContent = '0.0 cm';
    } else {
      verdictBanner.className = 'p-5 flex items-center justify-between text-white bg-rose-600 neon-out';
      verdictText.textContent = 'OUT';
      marginText.textContent = `+${data.distance_cm ?? 0} cm`;
    }

    matchStatusBadge.textContent = data.match_status || 'CONFIRMED';
    impactInfoText.textContent = `Frame #${data.impact_frame_idx ?? 0} (${Math.round(data.timestamp_ms || 0)}ms)`;

    // 스냅샷 이미지
    const snapshotWrapper = snapshotImg.parentElement;
    if (data.snapshot_url) {
      snapshotWrapper.classList.remove('hidden');
      snapshotImg.src = data.snapshot_url;
      snapshotImg.onclick = () => openImageModal(data.snapshot_url);
    } else {
      snapshotWrapper.classList.add('hidden');
      snapshotImg.removeAttribute('src');
      snapshotImg.onclick = null;
    }

    // AI 해설 정보
    const comm = data.commentary || {};
    summaryText.textContent =
      comm.one_line_summary ||
      (isUncertain ? '셔틀콕 착지 지점을 특정하지 못했습니다.' : isIN ? '베이스라인 인 판정입니다.' : '베이스라인 아웃 판정입니다.');
    commentaryText.textContent = comm.detailed || 'AI 해설이 제공되지 않았습니다.';
    engineTypeTag.textContent = comm.engine_type || 'LOCAL AI';

    // 피드백 UI 초기화
    feedbackButtons.classList.remove('hidden');
    feedbackCompletedMsg.classList.add('hidden');

    // 결과 카드 노출 및 스크롤
    resultCard.classList.remove('hidden');
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

    if (window.lucide) {
      window.lucide.createIcons();
    }
  }

  // 4. 피드백 제출
  btnFeedbackAgree.addEventListener('click', () => submitFeedback(true));
  btnFeedbackDisagree.addEventListener('click', () => submitFeedback(false));

  async function submitFeedback(agreed) {
    if (!currentRecordId) return;

    let comment = '';
    let userDecision = null;

    if (!agreed) {
      const reason = prompt('판정이 틀렸다고 생각하시는 이유를 간단히 적어주세요 (예: 라인에 살짝 걸쳤음):');
      if (reason === null) return; // 취소
      comment = reason;
      userDecision = verdictText.textContent === 'IN' ? 'OUT' : 'IN';
    }

    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          recordId: currentRecordId,
          agreed,
          user_decision: userDecision,
          comment,
        }),
      });

      const json = await res.json();
      if (json.success) {
        feedbackButtons.classList.add('hidden');
        feedbackCompletedMsg.classList.remove('hidden');
        loadHistory();
      }
    } catch (err) {
      alert(`피드백 제출 실패: ${err.message}`);
    }
  }

  // 5. 판정 이력 조회 및 렌더링
  btnRefreshHistory.addEventListener('click', loadHistory);

  async function loadHistory() {
    try {
      const res = await fetch('/api/judge/history?limit=10');
      const json = await res.json();
      if (!json.success) return;

      const items = json.data || [];
      historyCount.textContent = `${items.length}건`;

      if (items.length === 0) {
        historyList.innerHTML = '<p class="text-xs text-slate-500 text-center py-4">아직 판독 기록이 없습니다.</p>';
        return;
      }

      historyList.innerHTML = items
        .map((item) => {
          const decision = item.final_decision || 'UNCERTAIN';
          const isIN = decision === 'IN';
          const isUncertain = decision === 'UNCERTAIN';
          const badgeClass = isUncertain
            ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
            : isIN
              ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
              : 'bg-rose-500/20 text-rose-400 border-rose-500/30';
          const timeStr = new Date(item.created_at).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

          const feedbackBadge = item.feedback
            ? item.feedback.agreed
              ? '<span class="text-[10px] text-emerald-400">👍 인정</span>'
              : '<span class="text-[10px] text-rose-400">👎 불복(리뷰대기)</span>'
            : '';

          const safeId = escapeHtml(item.id);
          const safeDecision = escapeHtml(decision);
          const safeSummary = escapeHtml(item.commentary?.one_line_summary || item.video_filename);
          const safeDist = escapeHtml(item.distance_cm ?? 0);

          return `
            <div class="flex items-center justify-between p-2.5 rounded-xl bg-slate-950/60 border border-slate-800 hover:border-slate-700 cursor-pointer transition active:scale-[0.99]" onclick="window.viewHistoryDetail('${safeId}')">
              <div class="flex items-center gap-2.5">
                <span class="text-xs font-black px-2 py-0.5 rounded border ${badgeClass}">
                  ${safeDecision}
                </span>
                <div>
                  <p class="text-xs font-medium text-slate-200 truncate max-w-[170px]">${safeSummary}</p>
                  <p class="text-[10px] text-slate-500">${escapeHtml(timeStr)} · 오차 ${safeDist}cm</p>
                </div>
              </div>
              <div>${feedbackBadge}</div>
            </div>
          `;
        })
        .join('');
    } catch (err) {
      console.error('히스토리 로드 실패:', err);
    }
  }

  // 히스토리 항목 클릭 시 다시 결과 카드에 띄우기
  window.viewHistoryDetail = async (id) => {
    try {
      const res = await fetch(`/api/judge/${id}`);
      const json = await res.json();
      if (json.success && json.data) {
        renderResult(json.data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  // 모달 이미지 줌
  function openImageModal(url) {
    modalImg.src = url;
    imageModal.classList.remove('hidden');
  }
  btnCloseModal.addEventListener('click', () => {
    imageModal.classList.add('hidden');
  });
  imageModal.addEventListener('click', (e) => {
    if (e.target === imageModal) imageModal.classList.add('hidden');
  });

  // 로딩 헬퍼
  function showLoading(msg) {
    loadingStep.textContent = msg;
    loadingCard.classList.remove('hidden');
    resultCard.classList.add('hidden');
    loadingCard.scrollIntoView({ behavior: 'smooth' });
  }

  function hideLoading() {
    loadingCard.classList.add('hidden');
  }

  // 초기 히스토리 로드
  loadHistory();
});
