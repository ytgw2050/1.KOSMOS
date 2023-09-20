import xml.etree.ElementTree as ET
from geopy.distance import geodesic
import requests


class KOSMOS_localHospital:
    def __init__(self):
        # Parse the sample XML data into a list of AEDs
        self.parsed_aed_data = KOSMOS_localHospital.parse_xml_data(self.getKoreaAEDData())
        self.parsed_hospital_data = KOSMOS_localHospital.parse_xml_data(self.getKoreaHospitalData())
    
    def getKoreaAEDData(self):
        """
        # API URL
        url = "https://apis.data.go.kr/B552657/AEDInfoInqireService/getAedFullDown"

        # 파라미터 설정
        params = {
            "serviceKey": "***",
            "numOfRows": 5
        }

        # API 호출
        response = requests.get(url, params=params)
        
        return response.text
        """

        return open('local_hospital/data/AED_DATA.xml', 'r', encoding='utf-8').read()
    
    def getKoreaHospitalData(self):
        return open('local_hospital/data/HOSPITAL_DATA.xml', 'r', encoding='utf-8').read()

    # Parse the XML data and create a list of AED locations
    def parse_xml_data(xml_data):
        root = ET.fromstring(xml_data)
        center_list = []

        for item in root.find('.//items').iter('item'):
            lat = float(item.find('wgs84Lat').text)
            lon = float(item.find('wgs84Lon').text)
            center_list.append((lat, lon, item))

        return center_list

    # Calculate distance between two coordinates using geopy
    def calculate_distance(self, coord1, coord2):
        return geodesic(coord1, coord2).kilometers

    # Get AEDs within 1km radius of a given coordinate
    def get_supportCenter_in_radius(self, center_list, center_coord):
        nearby_aeds = []

        for supportCenter in center_list:
            supportCenter_coord = (supportCenter[0], supportCenter[1])
            distance = self.calculate_distance(center_coord, supportCenter_coord)
            if distance <= 1:
                nearby_aeds.append(supportCenter[2])

        return nearby_aeds