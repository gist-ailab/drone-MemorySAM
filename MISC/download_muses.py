import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# 설정
target_url = "https://muses.ethz.ch/MUSES_packages/"
download_folder = "/media/jemo/새 볼륨/dset/drone/DATA"

# 폴더 생성
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

def download_muses():
    print(f"Accessing {target_url}...")
    try:
        response = requests.get(target_url)
        response.raise_for_status() # 오류 발생 시 중단
    except Exception as e:
        print(f"접속 실패: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 모든 <a> 태그 중 파일 확장자가 포함된 링크 찾기
    # (서버 응답 형태에 따라 href 조건을 수정해야 할 수 있습니다)
    links = soup.find_all('a')
    
    for link in links:
        href = link.get('href')
        if not href or href.startswith('?') or href.startswith('/'):
            continue
            
        # 절대 경로 생성
        file_url = urljoin(target_url, href)
        file_name = os.path.join(download_folder, href)

        print(f"Downloading {href}...")
        
        # 실제 파일 다운로드
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    print("모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    download_muses()