"""
commentary.py
-------------
생성형 AI(Vision-Language Model & LLM)를 활용한
1) 착지 스냅샷 2차 시각 교차 검증 (VLM Verification)
2) 동호인 친화적 전문 해설 멘트 자동 생성 (AI Commentary) 모듈
"""

import json
import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class CommentaryEngine:
    """
    착지 순간 스냅샷과 1차 비전 물리 데이터를 종합하여
    2차 시각 검증 및 경기 해설 브리핑을 생성하는 엔진
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        """
        초기화 및 API 클라이언트 설정
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                print("[COMMENTARY] Google GenAI(Gemini) 클라이언트가 초기화되었습니다.")
            except ImportError:
                print("[WARN] 'google-genai' 패키지가 없습니다. 로컬 룰베이스 해설 모드로 동작합니다.")
            except Exception as e:
                print(f"[WARN] GenAI 초기화 실패: {e}. 로컬 모드로 전환합니다.")
        else:
            print("[INFO] GEMINI_API_KEY 미설정: 고정밀 로컬 룰베이스 해설 모드로 동작합니다.")

    def analyze_and_comment(
        self,
        snapshot_path: str,
        physical_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        스냅샷 이미지와 1차 물리 판정 결과를 바탕으로
        2차 시각 검증과 맞춤형 해설 리포트를 생성합니다.

        Args:
            snapshot_path: 호크아이 스냅샷 이미지 파일 경로
            physical_result: 1차 물리 판정 결과 (decision, distance_cm, status_detail 등)

        Returns:
            dict: {
                "final_decision": "IN" | "OUT",
                "match_status": "CONFIRMED" | "OVERRULED",
                "vlm_decision": "IN" | "OUT",
                "vlm_confidence": float,
                "one_line_summary": str,
                "commentary": str,
                "engine_type": "GEMINI_VLM" | "LOCAL_RULE_BASED"
            }
        """
        decision_1st = physical_result.get("decision", "UNCERTAIN")
        dist_cm = physical_result.get("distance_cm", 0.0)
        status_detail = physical_result.get("status_detail", "")
        contact_pt = physical_result.get("contact_point", (0, 0))

        # API 클라이언트가 준비되어 있고 파일이 존재하는 경우 실제 VLM 호출
        if self.client and os.path.exists(snapshot_path):
            try:
                from PIL import Image
                img = Image.open(snapshot_path)

                prompt = f"""
당신은 배드민턴 국제심판이자 재치 있는 공식 경기 해설위원 '스마트 선심'입니다.
제공된 이미지는 비전 추론 엔진이 포착한 셔틀콕 착지 순간의 호크아이 스냅샷입니다.
우측 상단에는 착지 지점이 2.5배 확대된 돋보기 인셋(IMPACT ZOOM)이 위치하며, 빨간색 가로선은 베이스라인 외곽 경계선(Outer Baseline)입니다.

[1차 비전 물리 데이터]
- 1차 물리 판정: {decision_1st} ({status_detail})
- 라인 외곽선과의 오차 거리: {dist_cm} cm
- 착지 접촉 좌표: {contact_pt}

[판정 규정]
배드민턴 규정상 셔틀콕의 코르크나 깃털이 베이스라인 외곽선에 1mm라도 닿거나 걸치면 무조건 IN이며, 완전히 벗어났을 때만 OUT입니다.

[요청 사항]
1. 돋보기 뷰의 십자선과 빨간 베이스라인 외곽선을 면밀히 관찰하여 시각적 In/Out을 2차 검증하십시오.
2. 1차 물리 판정과 일치하는지(CONFIRMED) 번복이 필요한지(OVERRULED) 결정하십시오.
3. 코트 위의 동호인들이 한눈에 직관적으로 이해할 수 있는 명쾌한 [한 줄 요약]을 작성하십시오.
4. 랠리의 긴장감과 착지 위치의 아슬아슬함을 생생하게 전하는 2~3문장의 [전문 해설 멘트]를 위트 있게 작성하십시오.

반드시 마크다운 코드블록(```json ... ```) 안에 유효한 JSON 형식으로만 응답해 주세요:
{{
  "vlm_decision": "IN" 또는 "OUT",
  "vlm_confidence": 0.95,
  "final_decision": "IN" 또는 "OUT",
  "match_status": "CONFIRMED" 또는 "OVERRULED",
  "one_line_summary": "명쾌한 한 줄 요약",
  "commentary": "생생하고 전문적인 해설 멘트"
}}
"""
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[img, prompt],
                )

                parsed = self._extract_json(response.text)
                if parsed:
                    parsed["engine_type"] = f"GEMINI_{self.model_name.upper()}"
                    return parsed

            except Exception as e:
                print(f"[WARN] VLM API 호출 중 예외 발생: {e}. 로컬 해설 엔진으로 대체합니다.")

        # API 미설정 또는 호출 실패 시 정교한 로컬 룰베이스 해설 생성
        return self._generate_fallback_commentary(physical_result)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """LLM 응답 텍스트에서 JSON 객체 파싱"""
        try:
            # ```json ... ``` 패턴 매칭
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            # 중괄호 직접 추출
            match2 = re.search(r"(\{.*\})", text, re.DOTALL)
            if match2:
                return json.loads(match2.group(1))
        except Exception:
            pass
        return None

    def _generate_fallback_commentary(self, physical_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        오프라인 환경에서도 동호인들이 실감 나게 즐길 수 있는
        상황별 지능형 룰베이스 해설 생성기
        """
        decision = physical_result.get("decision", "IN")
        status = physical_result.get("status_detail", "ON_LINE")
        dist = physical_result.get("distance_cm", 0.0)
        confidence = physical_result.get("confidence", 0.90)

        if decision == "IN":
            if status == "ON_LINE":
                summary = "라인에 살짝 걸쳤습니다! 극적인 IN 판정!"
                commentary = (
                    f"정말 아슬아슬한 승부였습니다! 셔틀콕 코르크가 백 바운더리 라인 끝자락을 아슬아슬하게 물고 들어갔습니다. "
                    f"수비 선수가 아웃을 확신하고 흘려보냈지만, 호크아이 확대 결과 확실한 IN으로 판명되었습니다!"
                )
            else:
                summary = "베이스라인 안쪽에 안전하게 안착한 명백한 IN입니다."
                commentary = (
                    f"깊숙하게 찔러 넣은 클리어 샷이 베이스라인 안쪽 코트에 정타로 꽂혔습니다. "
                    f"수비수의 타이밍을 완벽히 빼앗은 훌륭한 엔드라인 공략이었습니다."
                )
        else:  # OUT
            if dist <= 3.0:
                summary = f"라인을 불과 {dist}cm 벗어났습니다! 아쉬운 OUT!"
                commentary = (
                    f"아, 정말 손톱 하나 차이입니다! 셔틀콕이 베이스라인 외곽선에서 단 {dist}cm 벗어나며 착지했습니다. "
                    f"공격자의 과감한 딥 클리어 시도였으나 한 끗 차이로 바깥쪽에 떨어지며 점수는 수비 측으로 넘어갑니다."
                )
            elif dist <= 10.0:
                summary = f"베이스라인을 {dist}cm 지나쳐 떨어졌습니다. 확실한 OUT!"
                commentary = (
                    f"셔틀콕에 약간 힘이 실렸네요. 백 바운더리 라인을 {dist}cm 넘어가며 코트 밖으로 벗어났습니다. "
                    f"수비 선수의 정확한 라인 아웃 콜과 훌륭한 판단력이 빛을 발했습니다."
                )
            else:
                summary = f"엔드라인을 {dist}cm 크게 벗어난 확실한 OUT입니다."
                commentary = (
                    f"길게 뻗어나간 셔틀콕이 베이스라인을 {dist}cm 훌쩍 지나쳤습니다. "
                    f"주저 없이 흘려보낸 수비수의 완벽한 판단이 돋보였습니다."
                )

        return {
            "final_decision": decision,
            "match_status": "CONFIRMED",
            "vlm_decision": decision,
            "vlm_confidence": confidence,
            "one_line_summary": summary,
            "commentary": commentary,
            "engine_type": "LOCAL_RULE_BASED",
        }
