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
from navigation.library.SF_roadnetwork import RoadNetwork




class Navigation:
    def __init__(self, graph, intersection_dict_updated):
        self.graph = graph
        self.intersection_dict_updated = intersection_dict_updated

    def choice_purpose(self):
        start_point = 0
        end_point = 0
        while start_point == end_point:
            start_point = random.choices(list(self.intersection_dict_updated.keys()))
            end_point = random.choices(list(self.intersection_dict_updated.keys()))

        return start_point[0], end_point[0]

    def dijkstra(self, start, end, num_recommendations):
        num_recommendations=7
        paths = nx.shortest_simple_paths(self.graph, start, end)
        recommendations = []

        
        for _ in range(num_recommendations):
            try:
                path = next(paths)
                distance = nx.dijkstra_path_length(self.graph, start, end, weight='weight')
                recommendations.append((path, distance))
            except StopIteration:
                break

        return recommendations

    def draw_path(self, recommendations):
        c_l = ['r', 'b', 'yellow', 'g', 'black','gray','pink']
        pos = nx.fruchterman_reingold_layout(self.graph,seed=777,k = 2)

        for i, recommendation in enumerate(recommendations):
            plt.figure(figsize=(20, 20))
            nx.draw_networkx_edges(self.graph, pos)
            path, _ = recommendation
            nx.draw_networkx_edges(self.graph, pos, edgelist=[(path[j], path[j + 1]) for j in range(len(path) - 1)],
                                   edge_color=c_l[i], width=5, alpha=0.5)

            nx.draw_networkx_nodes(self.graph, pos, nodelist=[path[0]], node_color='green', node_size=100)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[path[-1]], node_color='red', node_size=100)

        plt.show()



class ENV():
    def __init__(self,data):
        self.data = data
        
        
    def make_road(self,center,radius):
        self.road_network = RoadNetwork(self.data, center, radius)
        self.combined_gdf, self.intersection_dict, self.road_dict = self.road_network.process_data()
        
    def all_predict(self):
        model_weights = {'1994744827':{0: '-', 8: '-', 9: 0.02, 15: 0.02}} 
        self.G, self.intersection_dict_updated = self.road_network.make_network_graph(self.intersection_dict, self.road_dict,model_weights)
    
    
    def navigation(self):
        start, end = self.navi.choice_purpose()
        recommendations = self.navi.dijkstra(start, end, num_recommendations=3)
        self.navi.draw_path(recommendations)
        
    def start_train(self,n,center,radius):
        self.make_road(center,radius)
        model_weights = {'-':{'-'}} # 모델 아무것도 없이 하기
        self.G, self.intersection_dict_updated = self.road_network.make_network_graph(self.intersection_dict, self.road_dict,model_weights)
        self.navi = Navigation(self.G, self.intersection_dict_updated)
        self.road_network.draw_graph(self.combined_gdf, self.intersection_dict_updated, self.G)
        for i in range(n):
            self.all_predict()
            self.navigation()

