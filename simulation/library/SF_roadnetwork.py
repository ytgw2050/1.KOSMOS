import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import split
from shapely.geometry import Point, LineString, MultiPoint
from tqdm import tqdm
import networkx as nx
from shapely.geometry import LineString, Polygon
from itertools import combinations
from shapely.geometry import GeometryCollection
import library.SF_wall as Wall

# warnings.filterwarnings('igxnore')
class RoadNetwork:
    def __init__(self, data, center, radius):
        self.data = data
        self.center = center
        self.radius = radius
        self.RN_road = []

    def find_all_indices_range(self):
        range_road = []
        center_point = Point(self.center)
        center_x, center_y = center_point.x, center_point.y  
        radius_squared = self.radius ** 2

        for road,rn in zip(self.data['geometry'].tolist(),self.data['RN'].tolist()):
            try:
                for point in road.coords:
                    dist_squared = (point[0] - center_x) ** 2 + (point[1] - center_y) ** 2  
                    if dist_squared <= radius_squared:
                        range_road.append(road) 
                        self.RN_road.append(rn)
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

            if intersection and intersection.geom_type == 'Point':
                point_tuple = (intersection.x , intersection.y )
                point_name = str(intersection.x)[:5] + str(intersection.y)[:5]

                if point_name not in intersection_dict:
                    intersection_dict[point_name] = {'road_position': point_tuple, 'road_list': {}}

                intersection_dict[point_name]['road_list'].setdefault(line1_index, '-')
                intersection_dict[point_name]['road_list'].setdefault(line2_index, '-')

        return intersection_dict, road_dict

    def split_roads(self, target_data, intersection_dict):
        split_roads = []
        
        center = Point(self.center)
        range_bbox = center.buffer(self.radius)
        filtered_gdf = gpd.GeoDataFrame(geometry=target_data.intersection(range_bbox))
        
        def split_and_append(road):
            segments = road.geometry if road.geometry.geom_type != 'LineString' else [road.geometry]
            for line in segments:
                positions = [item['road_position'] for item in list(intersection_dict.values()) if item['road_position'] in line.coords]
                if positions:
                    geometries = split(line, MultiPoint(positions))
                    if isinstance(geometries, GeometryCollection):
                        for geom in geometries.geoms:
                            split_roads.append(geom)

                    else:
                        split_roads.extend(geometries)

#                     split_roads.extend(split(line, MultiPoint(positions)))
        
        filtered_gdf.apply(split_and_append, axis=1)

        return split_roads
    
    def process_data(self):
        target_data = self.find_all_indices_range()
        intersection_dict, road_dict = self.find_intersections(target_data)
        new_roads = self.split_roads(target_data, intersection_dict)

        intersection_dict, road_dict = self.find_intersections(gpd.GeoSeries(new_roads))
        combined_gdf = gpd.GeoDataFrame(geometry=new_roads)

        
        buffer_distance = 10
        self.combined_gdf_buffered = combined_gdf.copy()
        self.combined_gdf_buffered['geometry'] = combined_gdf.buffer(buffer_distance)

        self.intersection_dict = intersection_dict
        self.road_dict = road_dict

        return self.combined_gdf_buffered, intersection_dict, road_dict



    def predict_network(self, intersection_dict,model_weights,target_model):
        def update_weights(model_weights, new_values):
            keys = list(model_weights.keys())  # keys를 리스트 형태로 가져옴
            for key in keys:
                sub_keys = list(model_weights[key].keys())  # sub keys를 리스트 형태로 가져옴
                for sub_key, new_value in zip(sub_keys, new_values):
                    model_weights[key][sub_key] = new_value  # 새로운 값을 설정함

            return model_weights


        
        G = nx.Graph()
        
        
        if target_model == None:
            model_weights = {'199473444827':{'199474234827': '-', '192474423827': '-', '199473234827': '-'}} 

        else:
            org_ = {target_model : intersection_dict[target_model]['road_list']}
            model_weights = update_weights(org_,model_weights)
        for node in intersection_dict.keys():
            G.add_node(node)

        for node1, node2 in combinations(intersection_dict.keys(), 2):
            common_roads = set(intersection_dict[node1]['road_list']).intersection(set(intersection_dict[node2]['road_list']))

            if common_roads:
                common_value = common_roads.pop()

                point1 = Point(intersection_dict[node1]['road_position'])
                point2 = Point(intersection_dict[node2]['road_position'])
                distance = point1.distance(point2)

                weight1 = model_weights.get(node1, {}).get(common_value, 1)
                intersection_dict[node1]['road_list'][common_value] = distance * weight1
                
                G.add_edge(node1, node2, weight=distance * weight1)
        return G , intersection_dict
    
    def draw_graph(self, combined_gdf, intersection_dict, G):
        positions = [item['road_position'] for item in list(intersection_dict.values())]
        combined_gdf.plot(figsize=(20,20))
        for point in tqdm(positions):
            plt.scatter(*point, color='red',s = 1000)
        plt.show()
        
        plt.figure(figsize=(20, 20))
        # pos = {node: intersection_dict[node]['road_position'] for node in G.nodes()}
        pos = nx.fruchterman_reingold_layout(G,seed=777,k = 2)
        nx.draw_networkx(G, pos=pos, with_labels=True, font_size=10, node_size=2000)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        
        plt.show()


    def make_road(self):
        union_polygon = self.combined_gdf_buffered.unary_union
        outer_lines = []

        if isinstance(union_polygon, Polygon):
            union_polygon = [union_polygon]

        for poly in union_polygon:
            x, y = poly.exterior.coords.xy
            outer_lines.extend([LineString([(x[i], y[i]), (x[i+1], y[i+1])]) for i in range(len(x)-1)])

            for interior in poly.interiors:
                x, y = interior.coords.xy
                outer_lines.extend([LineString([(x[i], y[i]), (x[i+1], y[i+1])]) for i in range(len(x)-1)])

        return [Wall.Wall(line.coords[0], line.coords[-1]) for line in outer_lines]
