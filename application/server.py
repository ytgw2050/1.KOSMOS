from flask import Flask, request, jsonify
from flask_cors import CORS

import osmnx as ox
from navigation.navigation import KOSMOS_Navigation
from local_hospital.local_hospital_loc import KOSMOS_localHospital


KosmosNavi = KOSMOS_Navigation()
KosmosLocalHospital = KOSMOS_localHospital()

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
	return "본 API는 GET 형식 요청을 받지 않습니다."

@app.route('/roadPath', methods=['POST'])
def getRoadPathData():
    requestData = request.json

    location = ((requestData['location'])[0], (requestData['location'])[1])
    graph = ox.graph_from_point(location, dist=(requestData['distance']), network_type='drive')

    node_dict = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
    edge_dict = {(u, v): data for u, v, key, data in graph.edges(keys=True, data=True)}

    geometry_data = {}
    for key in edge_dict.keys():
            if 'geometry' in edge_dict[key]:
                geometry_data['{}-{}'.format(key[0], key[1])] = edge_dict[key]['geometry'].coords[:]

    resultData = {
        "node": node_dict,
        "edge": list(edge_dict.keys()),
        "edge_geometry": geometry_data,
    }

    return jsonify(resultData), 200

@app.route('/navigation/data', methods=['GET'])
def getNaviRoadData():
    resultData = {
         "data": KosmosNavi.getIntersectionPointData()
    }

    return jsonify(resultData), 200

@app.route('/navigation/path', methods=['POST'])
def getNaviRoutePathData():
    requestData = request.json

    resultData = {
         "data": KosmosNavi.getPath(requestData['startPoint'], requestData['destinationPoint'])
    }

    return jsonify(resultData), 200


@app.route('/navigation/coords', methods=['POST'])
def convertCoordsSystem(): #행정안전부 데이터 내 좌표계를 구글과 같은 일반 좌표계로 변환
    requestData = request.json
    print(requestData['data'])

    resultData = {
        "data": KosmosNavi.convertCoordsSystem(requestData['data']).tolist()
    }
    print(resultData)

    return jsonify(resultData), 200


@app.route('/nearby_center/aed', methods=['GET'])
def get_nearby_aeds():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))
    center_coord = (lat, lon)

    nearby_aeds = KosmosLocalHospital.get_supportCenter_in_radius(KosmosLocalHospital.parsed_aed_data, center_coord)

    aed_list_json = []
    for aed in nearby_aeds:
        aed_dict = {
            "id": aed.find('rnum').text,
            "buildAddress": aed.find('sido').text + " " + aed.find('buildAddress').text,
            "buildPlace": aed.find('buildPlace').text,
            "manager": aed.find('manager').text,
            "tel": aed.find('managerTel').text,
            "lat": aed.find('wgs84Lat').text,
            "lon": aed.find('wgs84Lon').text,
        }
        aed_list_json.append(aed_dict)

    response = {
        "data": aed_list_json,
        "dataAmount": len(aed_list_json)
    }

    return jsonify(response)

@app.route('/nearby_center/hospital', methods=['GET'])
def get_nearby_hospitals():
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))
    center_coord = (lat, lon)

    nearby_hospitals = KosmosLocalHospital.get_supportCenter_in_radius(KosmosLocalHospital.parsed_hospital_data, center_coord)

    hospital_list_json = []
    for hospital in nearby_hospitals:
        aed_dict = {
            "id": hospital.find('rnum').text,
            "buildAddress": hospital.find('dutyAddr').text,
            "buildPlace": hospital.find('dutyName').text,
            "tel": hospital.find('dutyTel1').text,
            "emclsType": hospital.find('dutyEmclsName').text,
            "lat": hospital.find('wgs84Lat').text,
            "lon": hospital.find('wgs84Lon').text,
        }
        hospital_list_json.append(aed_dict)

    response = {
        "data": hospital_list_json,
        "dataAmount": len(hospital_list_json)
    }

    return jsonify(response)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)