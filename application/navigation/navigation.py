import navigation.library.SF_env as SFm
import navigation.library.SF_navigation as Nav

import matplotlib.animation as animation
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

import pyproj
import pandas as pd


class KOSMOS_Navigation:
    def __init__(self):
        self.data = gpd.read_file('navigation/data/Z_KAIS_TL_SPRD_MANAGE_11000.shp', encoding='euc-kr')

        center = [199430.0,448360.0]
        radius = 1000

        self.road_network = Nav.RoadNetwork(self.data, center, radius)
        combined_gdf, self.intersection_dict, self.road_dict = self.road_network.process_data()
    
    def getPath(self, startPoint, endPoint):
        model_weights = {'1994744827':{0: '-', 8: '-', 9: 0.02, 15: 0.02}} 


        G, intersection_dict_updated = self.road_network.make_network_graph(self.intersection_dict, self.road_dict, model_weights)


        navi = Nav.Navigation(G, intersection_dict_updated)
        print(navi.choice_purpose())

        recommendations = navi.dijkstra(startPoint, endPoint, num_recommendations=3)

        return recommendations

    
    def getIntersectionPointData(self):
        return self.intersection_dict
    
    def convertCoordsSystem(self, coordsData):
        source = pyproj.Proj(init="epsg:5174")
        target = pyproj.Proj(init="epsg:4326")

        df = pd.DataFrame(coordsData)
        coord = np.array(df)

        fx, fy = pyproj.transform(source, target, coord[:, 0], coord[:, 1])
        return np.dstack([fx, fy])[0]

