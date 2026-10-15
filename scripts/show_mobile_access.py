"""
show_mobile_access.py
---------------------
체육관 코트 현장에서 스마트폰으로 AI Hawk-Eye Lite에 접속할 수 있는
로컬 IP 주소 및 터널링(ngrok / Cloudflare) 안내 출력 스크립트
"""

import socket
import sys

def get_local_ip():
    """현재 로컬 네트워크(Wi-Fi) IP 주소 탐색"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 외부 DNS 서버 연결 시도로 실제 발신 네트워크 인터페이스 IP 획득
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def main():
    port = 4000
    local_ip = get_local_ip()

    print("=" * 65)
    print("🏸 [AI Hawk-Eye Lite] 스마트폰 모바일 접속 가이드")
    print("=" * 65)
    print(f"\n[방법 1] 체육관 동일 Wi-Fi에 연결된 경우 (가장 빠름):")
    print(f" -> 스마트폰 브라우저에서 접속: http://{local_ip}:{port}/")
    print(f" -> 관리자(오심 리뷰 큐) 접속:  http://{local_ip}:{port}/admin.html")

    print(f"\n[방법 2] 스마트폰이 LTE/5G 모바일 데이터인 경우 (터널링):")
    print(" -> 무료 Cloudflare Tunnel 사용:")
    print(f"    npx cloudflared tunnel --url http://localhost:{port}")
    print(" -> 또는 ngrok 사용:")
    print(f"    ngrok http {port}")
    print(" -> 터널링 실행 후 출력되는 https://xxx.trycloudflare.com 또는 https://xxx.ngrok-free.app 주소로 접속하세요!")
    print("\n" + "=" * 65)

if __name__ == "__main__":
    main()
