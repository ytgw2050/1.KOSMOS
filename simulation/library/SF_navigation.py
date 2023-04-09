import matplotlib.pyplot as plt
import networkx as nx
import random
from library.SF_roadnetwork import RoadNetwork
import numpy as np




class Navigation:
    def __init__(self, graph, intersection_dict,shop_dict):
        self.graph = graph
        self.intersection_dict = intersection_dict
        self.shop_dict = shop_dict
        self.save_rec = []
        
    def choice_purpose_shop(self):
        start_point = 0
        end_point = 0
        while start_point == end_point:
            start_point = random.choices(list(self.shop_dict.keys()))
            end_point = random.choices(list(self.shop_dict.keys()))

        return start_point[0], end_point[0]
    
    def find_closest_key(self, x, y):
        data = self.intersection_dict
        closest_distance = float('inf')  # 초기 거리는 무한대로 설정
        closest_key = None  # 가장 가까운 key

        for key, value in data.items():
            position = value['road_position']
            distance = np.linalg.norm(np.array(position) - np.array([x, y]))  # Euclidean distance 계산

            if distance < closest_distance:
                closest_distance = distance
                closest_key = key

        return closest_key
    
    def choice_purpose_intersection(self):
        start_intersection_point = 0
        end_intersection_point = 0
        while start_intersection_point == end_intersection_point:
            start_shop_point, end_shop_point = self.choice_purpose_shop()
            start_intersection_point = self.find_closest_key(self.shop_dict[start_shop_point]['location'][0], self.shop_dict[start_shop_point]['location'][1])
            end_intersection_point = self.find_closest_key(self.shop_dict[end_shop_point]['location'][0], self.shop_dict[end_shop_point]['location'][1])
        return start_shop_point, end_shop_point, start_intersection_point, end_intersection_point
    
  
    def dijkstra(self, start, end, num_recommendations):
        num_recommendations = 3
        paths = nx.shortest_simple_paths(self.graph, start, end, weight='weight')
        recommendations = []

        for _ in range(num_recommendations):
            try:
                path = next(paths)
                distance = sum(self.graph.edges[path[i], path[i + 1]]['weight'] for i in range(len(path) - 1))
                recommendations.append((path, distance))
            except StopIteration:
                break

        self.save_rec.append(recommendations)
        return recommendations


    def draw_path(self, recommendations,intersection_dict,G):
        c_l = ['r', 'b', 'yellow', 'g', 'black','gray','pink']
        # pos = nx.fruchterman_reingold_layout(self.graph,seed=777,k = 2)
        pos = {node: intersection_dict[node]['road_position'] for node in G.nodes()}

        for i, recommendation in enumerate(recommendations):
            plt.figure(figsize=(20, 20))
            nx.draw_networkx_edges(self.graph, pos)
            path, _ = recommendation
            nx.draw_networkx_edges(self.graph, pos, edgelist=[(path[j], path[j + 1]) for j in range(len(path) - 1)],
                                   edge_color=c_l[i], width=5, alpha=0.5)

            nx.draw_networkx_nodes(self.graph, pos, nodelist=[path[0]], node_color='green', node_size=100)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[path[-1]], node_color='red', node_size=100)

        plt.show()


