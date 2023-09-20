from flask import Flask, request, jsonify
from flask_cors import CORS

import library.SF_env
import library.shop_dict
import geopandas as gpd
import threading

app = Flask(__name__)
CORS(app)

data = gpd.read_file('data/Z_KAIS_TL_SPRD_MANAGE_11000.shp', encoding='euc-kr')
shopDictData = library.shop_dict.getShopDict(data)

envVariableMap = {
     "itaewon": library.SF_env.ENV(data, [199430.0,448360.0], 500, shop_dict=shopDictData),
     "gangnam": library.SF_env.ENV(data, [203500.0,446000.0], 500, shop_dict=shopDictData),
     "gwanghwamun": library.SF_env.ENV(data, [198500.0,451500.0], 500, shop_dict=shopDictData),
     "myeongdong": library.SF_env.ENV(data, [201000.0,455000.0], 500, shop_dict=shopDictData)
}

def runSimulation(envName):
    for i in range(5000):
        library.SF_env.Trian(envVariableMap[envName], 2, False, 1, "True", 1)

        with open(envName+".txt", "w") as f:
            f.write(str([[x.position] for x in envVariableMap[envName].pedestrians]))
        f.close()
    
     
for envName in envVariableMap:
    train_thread = threading.Thread(target=runSimulation, args=(envName,))
    train_thread.start()

@app.route("/")
def hello():
	return "본 API는 GET 형식 요청을 받지 않습니다."

@app.route('/data/pedestrian', methods=['GET'])
def getRoadPathData():
    location = request.args.get('location')

    resultData = {}

    if location in envVariableMap:
        try:
            with open(location+".txt", "r") as f:
                resultData = {
                    "location": f.read()
                }
        except:
            resultData = {"message": "아직 시뮬레이션이 동작하지 않았습니다. 잠시 후 시도해주세요."}
        
        return jsonify(resultData), 200
    else:
        resultData = {"message": "잘못된 요청입니다."} 

        return jsonify(resultData), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)