# 데이터 분석 및 처리
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import Counter

# 시각화
import folium
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 지리 데이터 처리
# from osgeo import osr
import pyproj

# 기타 라이브러리
import random
from tqdm import tqdm
import pickle
import warnings
from tensorflow.keras.models import load_model

# 사용자 정의 모듈
import library.SF_env as kos
import library.SF_navigation as Nav

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")

import requests
import time
from collections import defaultdict
import tensorflow as tf

# 글로벌로 Proj 객체를 생성하여 재사용
P_WGS84 = pyproj.Proj(init="epsg:4326")
P_KTM = pyproj.Proj(init="epsg:5181")


def moving_average(lst):
    window=10000
    ret = []
    for i in range(len(lst)):
        if i < window:
            ret.append(sum(lst[:i+1])/(i+1))
        else:
            ret.append(sum(lst[i-window+1:i+1])/window)
    return ret


def save_data(target_list, data_list, type_train, tm,episode):
    for target, data in zip(target_list, data_list):
        with open("data_set/{}_{}_{}.pkl".format(target[type_train],tm,episode), 'wb') as f:
            pickle.dump(data, f)
            
            
def load_variable(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def plot_data(data_list, colors=['r', 'b', 'g']):
    plt.figure(figsize=(20,10))
    for data, color in zip(data_list, colors):
        plt.plot(moving_average(data), c=color)
    



def project_array(coord, p1, p2):  # coord Type: [X, Y]
    fx, fy = pyproj.transform(p1, p2, coord[:, 0], coord[:, 1])
    return np.dstack([fx, fy])[0]


def ktm_to_wgs84(easting, northing):
    coord = np.array([[easting, northing]])
    result = project_array(coord, P_KTM, P_WGS84)
    return result[0][1], result[0][0]  # Latitude, Longitude


def wgs84_to_ktm(lat, lon):
    coord = np.array([[lon, lat]])  # 주의: 입력은 [Longitude, Latitude] 순서입니다
    result = project_array(coord, P_WGS84, P_KTM)
    return result[0][0], result[0][1]  # Easting, Northing


def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5


def get_nearby_places(latitude, longitude, api_key, radius):
    endpoint_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        'location': f'{latitude},{longitude}',
        'radius': radius,
        'type': 'store',
        'key': api_key
    }
    results = []
    while True:
        response = requests.get(endpoint_url, params=params)
        if response.status_code == 200:
            data = response.json()
            results.extend(data.get('results', []))
            
            # next_page_token이 있으면 다음 페이지의 결과를 요청
            next_page_token = data.get('next_page_token')
            if next_page_token:
                params['pagetoken'] = next_page_token
                
                # Google 문서에 따르면 next_page_token을 사용하기 전에 약간의 지연이 필요
                time.sleep(2)
            else:
                break
        else:
            print("Error:", response.status_code)
            break
    return results


def get_shops_by_street_name(api_key, street_name, radius, type='store'):
    endpoint_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    query = f"{type} in {street_name}"
    params = {
        'query': query,
        'radius': radius,
        'key': api_key
    }
    response = requests.get(endpoint_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('results', [])
    else:
        print("Error:", response.status_code)
        return []

def snap_store_to_nearest_wall(store, walls):
    min_distance = float('inf')
    snapped_point = None

    for wall in walls:
        distance = wall.distance_to(store)
        if distance < min_distance:
            min_distance = distance
            snapped_point = wall.closest_point_to(store)

    return snapped_point


def distance(loc1, loc2):
    return np.sqrt((loc1['lat'] - loc2['lat'])**2 + (loc1['lng'] - loc2['lng'])**2)

def compute_centroid(locations):
    lats = [loc['lat'] for loc in locations]
    lngs = [loc['lng'] for loc in locations]
    return {'lat': sum(lats) / len(lats), 'lng': sum(lngs) / len(lngs)}

def group_stores(stores, threshold_distance=0.0001): # 여기서 threshold_distance는 약 10m입니다.
    visited = set()
    groups = []

    for idx, store in enumerate(stores):
        if idx in visited:
            continue
        
        group = [store]
        stack = [store]
        visited.add(idx)

        while stack:
            current_store = stack.pop()
            current_location = current_store['geometry']['location']

            for i, other_store in enumerate(stores):
                if i in visited:
                    continue

                other_location = other_store['geometry']['location']
                if distance(current_location, other_location) < threshold_distance:
                    stack.append(other_store)
                    group.append(other_store)
                    visited.add(i)

        if group:
            groups.append(group)

    return groups

def get_group_average(group):
    avg_rating = np.mean([shop.get('rating', 0) for shop in group])
    centroid = compute_centroid([shop['geometry']['location'] for shop in group])
    return {'name': 'Grouped', 'geometry': {'location': centroid}, 'rating': avg_rating}


# 이후 상점 시각화 코드에서 shops 대신 averaged_stores 사용

def find_closest_key(data, x, y):
    closest_distance = float('inf')  # 초기 거리는 무한대로 설정
    closest_key = None  # 가장 가까운 key

    for key, value in data.items():
        position = value['road_position']
        distance = np.linalg.norm(np.array(position) - np.array([x, y]))  # Euclidean distance 계산

        if distance < closest_distance:
            closest_distance = distance
            closest_key = key

    return closest_key



from shapely.geometry import Polygon, Point

def getShopDict(data):
    P_WGS84 = pyproj.Proj(init="epsg:4326")
    P_KTM = pyproj.Proj(init="epsg:5181")

    org_easting = 199430.0
    org_northing = 448360.0
    center = [org_easting, org_northing]
    radius = 100
    env1 = kos.ENV(data, center, radius)
    env1.make_road()

    # 벽 객체들로부터 폴리곤을 생성합니다
    polygon_points = [wall.start.tolist() for wall in env1.walls]  # 벽의 시작점들을 가져옵니다
    polygon = Polygon(polygon_points)

    # 폴리곤을 20만큼 확장합니다
    expanded_polygon = polygon.buffer(30)

    api_key = 'AIzaSyD3Nldf8so7fxMsGCdVc-teOHZ3N-r22DA'


    lat, lng = ktm_to_wgs84(org_easting, org_northing)

    plt.figure(figsize=(20, 20))

    for wall in env1.walls:
        plt.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]], 'k')


    for wall in env1.walls:
        plt.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]], 'k')

    results = get_nearby_places(lat, lng, api_key, radius+100)

    shopss = []
    for place in tqdm(results):
        name = place['name']
        location = place['geometry']['location']
        lat, lng = location['lat'], location['lng']
        easting, northing = wgs84_to_ktm(lat, lng)
        point = Point(easting, northing)
        
        if expanded_polygon.contains(point):
            plt.scatter(easting, northing,s= 100)
            new_location = snap_store_to_nearest_wall(np.array([easting, northing]), env1.walls)
            if new_location is not None:
                shopss.append(place)
                plt.scatter(*new_location,s= 300, c='g') 
                plt.plot([easting,new_location[0]],[northing,new_location[1]])

    #plt.show() 




    shop_dict = {}

    for wall in env1.walls:
        plt.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]], 'k')

    grouped_stores = group_stores(shopss)
    averaged_stores = [get_group_average(group) for group in grouped_stores]
    ii = -1
    for place in tqdm(averaged_stores):
        ii += 1
        name = place['name']
        location = place['geometry']['location']
        lat, lng = location['lat'], location['lng']
        easting, northing = wgs84_to_ktm(lat, lng)
        point = Point(easting, northing)
        
        if expanded_polygon.contains(point):
            plt.scatter(easting, northing)
            new_location = snap_store_to_nearest_wall(np.array([easting, northing]), env1.walls)
            if new_location is not None:
                shop_dict[ii] = {'rating' : place['rating'] , 'location' : [new_location[0], new_location[1]]}
                locc = env1.intersection_dict[find_closest_key(env1.intersection_dict, new_location[0], new_location[1])]['road_position']


    print(shop_dict)
    return shop_dict