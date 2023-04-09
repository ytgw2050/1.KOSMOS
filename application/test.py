import requests
import xml.etree.ElementTree as ET

# API URL
url = "https://apis.data.go.kr/B552657/AEDInfoInqireService/getAedFullDown"

# 파라미터 설정
params = {
    "serviceKey": "2uO2vJc2ZEiCy887GWfADY97XtQppTDYGzRLfMQ2iEg5dIyPN0W5Z0WiK4opKpAIhvLskyv/p6cVtxqDnIudFg==",
    "numOfRows": 5
}

# API 호출
response = requests.get(url, params=params)

# XML 파싱
root = ET.fromstring(response.text)

# 나머지 코드는 위에서 제공한 예시 코드와 유사합니다
print(root)
print(response.text)