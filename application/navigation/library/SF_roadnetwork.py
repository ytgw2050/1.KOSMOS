import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.ops import split
from shapely.geometry import Point, LineString, MultiPoint
from tqdm import tqdm
import networkx as nx
from itertools import combinations
from operator import itemgetter
import random
import warnings

# warnings.filterwarnings('igxnore')

class RoadNetwork:
    def __init__(self, data, center, radius):
        self.data = data
        self.center = center
        self.radius = radius

    def find_all_indices_range(self):
        range_road = []
        center_point = Point(self.center)
        center_x, center_y = center_point.x, center_point.y  # center의 x와 y 좌표 분리
        radius_squared = self.radius ** 2

        for i, road in enumerate(self.data['geometry'].tolist()):
            try:
                for point in road.coords:
                    dist_squared = (point[0] - center_x) ** 2 + (point[1] - center_y) ** 2  
                    if dist_squared <= radius_squared:
                        range_road.append(road) 
                        break
            except:
                pass
            
        return gpd.GeoSeries(range_road)
    


    def find_intersections(self, target_data):
        intersection_dict = {}
        road_dict = {}
        for inlo, line in enumerate(target_data):
            road_dict[inlo] = line

        for line1_index, line2_index in combinations(list(range(len(target_data))), 2):
            intersection = road_dict[line1_index].intersection(road_dict[line2_index])

            if intersection.geom_type == 'Point':
                point_tuple = (intersection.x , intersection.y )
                point_name = str(intersection.x)[:5] + str(intersection.y)[:5] # 고유 번호 생성

                if point_name not in intersection_dict:
                    intersection_dict[point_name] = {}
                    intersection_dict[point_name]['road_position'] = (point_tuple)
                    intersection_dict[point_name]['road_list'] = {}

                if line1_index not in intersection_dict[point_name]['road_list']:
                    intersection_dict[point_name]['road_list'][line1_index] = '-'
                if line2_index not in intersection_dict[point_name]['road_list']:
                    intersection_dict[point_name]['road_list'][line2_index] = '-'

        return intersection_dict, road_dict


    def split_roads(self, target_data, intersection_dict):
        split_roads = []
        
        center = Point(self.center)
        range_bbox = center.buffer(self.radius)
        filtered_gdf = gpd.GeoDataFrame(geometry=target_data.intersection(range_bbox))
        
        for road in filtered_gdf.itertuples(index=False):
            segments = road.geometry
            if road.geometry.geom_type == 'LineString': # for 문 돌리기 위해서
                segments = [segments]
            for line in segments:
                positions = [item['road_position'] for item in list(intersection_dict.values())]
                intersection_points = [Point(point) for point in positions if point in line.coords]
                if intersection_points:
                    split_roads.extend(split(line, MultiPoint(intersection_points)))
        
        return split_roads

    def process_data(self):
        target_data = self.find_all_indices_range()
        
        intersection_dict , road_dict = self.find_intersections(target_data)
        new_roads = self.split_roads(target_data, intersection_dict)
        
        intersection_dict, road_dict = self.find_intersections(gpd.GeoSeries(new_roads))
        combined_gdf = gpd.GeoDataFrame(geometry=new_roads)
        
        return combined_gdf, intersection_dict, road_dict

    def make_network_graph(self, intersection_dict, road_dict, model_weights = None):
        G = nx.Graph()

        for node in intersection_dict.keys():
            G.add_node(node)
        for node1, node2 in tqdm(combinations(intersection_dict.keys(), 2)):
            road_list1 = list(intersection_dict[node1]['road_list'].keys())
            road_list2 = list(intersection_dict[node2]['road_list'].keys())
            common_roads = set(road for road in road_list1) & set(road for road in road_list2)

            if common_roads:
                common_value = common_roads.pop()

                point1 = Point(intersection_dict[node1]['road_position'])
                point2 = Point(intersection_dict[node2]['road_position'])
                distance = point1.distance(point2)

                weight1 = model_weights.get(node1, {}).get(common_value, 1)
                weight2 = model_weights.get(node2, {}).get(common_value, 1)

                intersection_dict[node1]['road_list'][common_value] = distance * weight1
                intersection_dict[node2]['road_list'][common_value] = distance * weight2
                
                G.add_edge(node1, node2, weight=distance * weight1)

        # 연결된 간선이 2개 미만인 노드를 제거하는 과정 추가
        to_remove = [node for node, degree in G.degree() if degree < 2]
        G.remove_nodes_from(to_remove)

        return G , intersection_dict

    def draw_graph(self, combined_gdf, intersection_dict, G):
        positions = [item['road_position'] for item in list(intersection_dict.values())]
        combined_gdf.plot(figsize=(20,20))
        for point in tqdm(positions):
            plt.scatter(*point, color='red')
        plt.show()
        
        plt.figure(figsize=(20, 20))
        pos = nx.fruchterman_reingold_layout(G,seed=777,k = 2)
        nx.draw_networkx(G, pos=pos, with_labels=True, font_size=3, node_size=20)
        plt.show()


