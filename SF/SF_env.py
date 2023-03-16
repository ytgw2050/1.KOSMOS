# 데이터 분석 및 가공
import numpy as np
import networkx as nx

# 시각화
from matplotlib import animation
import matplotlib.pyplot as plt
import random

# 데이터 분석
from tqdm import tqdm 

# 라이브러리
import random
import warnings
from collections import Counter

# 사용자 정의 모듈
from SF import SF_navigation as Navigation
from SF import SF_roadnetwork as RN
from SF import SF_pedestrian as PD
from SF import SF_agent as AGENT



warnings.filterwarnings('ignore')

class ENV:
    def __init__(self,data, center, radius,shop_dict = None):
        self.road_network = RN.RoadNetwork(data, center, radius)
        self.combined_gdf, self.intersection_dict, self.road_dict = self.road_network.process_data()
        self.G ,self.intersection_dict= self.road_network.predict_network(self.intersection_dict, model_weights = None,target_model = None)
        self.pedestrians = []
        self.make_road()
        self.num_pedestrians = 1
        self.navi = Navigation.Navigation(self.G,self.intersection_dict,shop_dict)

        self.intercection_point_list = {}
        self.state_list_100 = []
        self.count_pedestrains = []
        self.crowd_pressure_list_velocity = []

        self.end_time_list = []
        self.shop_dict = shop_dict
        


# 환경 제작
    def make_road(self): # 완료
        self.walls =  self.road_network.make_road()

    def make_pedestrians(self):
        for _ in range(self.num_pedestrians):
            recommendations = False
            while recommendations == False:
                start_point, end_point,start_intersection_point, end_intersection_point = self.navi.choice_purpose_intersection()
                recommendations = self.navi.dijkstra(start_intersection_point, end_intersection_point, num_recommendations=3)
            intersection_dict_recommend_0 = self.intersection_dict[recommendations[0][0][0]]['road_position']
            x = intersection_dict_recommend_0[0]
            y = intersection_dict_recommend_0[1]

            # x = self.shop_dict[start_point]['location'][0] # 상점 생성했을떄 벽에 붙어서 생성되는것 해결후 쓰기
            # y = self.shop_dict[start_point]['location'][1] #

            intersection_dict_recommend_1 = self.intersection_dict[recommendations[0][0][1]]['road_position']
            x_n = intersection_dict_recommend_1[0]
            y_n = intersection_dict_recommend_1[1]

            pedestrian = PD.Pedestrian([random.random() + x ,random.random() + y ],[random.random()+x_n,random.random()+y_n], random.random()*5+3, 0.5, 1, 1, 'r')
            pedestrian.target = 0
            pedestrian.color = 'r'
            self.pedestrians.append(pedestrian)
            pedestrian.recommendation = recommendations
            pedestrian.start_intersection_point = start_intersection_point
            pedestrian.end_intersection_point = end_intersection_point
            pedestrian.start_point = start_point
            pedestrian.end_point = end_point

    # def make_model(self):

# 환경 상태 정의 (보상)
    # 압력
    def calculate_density(self, R):
        if len(self.pedestrians) == 0:
            return np.array([0])  

        density = np.zeros(len(self.pedestrians))
        for i, p_i in enumerate(self.pedestrians):
            for j, p_j in enumerate(self.pedestrians):
                if i != j:
                    distance = np.linalg.norm(p_i.position - p_j.position)
                    local_density = np.exp(-(distance)**2 / (R**2)) # 식이 좀 이상하다
                    density[i] += local_density

        density /= np.pi * (R**2)

        density[np.isnan(density)] = 0   

        return density

    def calculate_velocity_variance(self):
        if len(self.pedestrians) == 0: 
            return np.array([0, 0])

        velocities = np.array([p.velocity for p in self.pedestrians])
        mean_velocity = np.mean(velocities, axis=0)
        velocity_variance = np.var(velocities - mean_velocity, axis=0)

        velocity_variance[np.isnan(velocity_variance)] = 0  

        return velocity_variance

    def calculate_crowd_pressure(self, R):
        density = self.calculate_density(R)
        velocity_variance = self.calculate_velocity_variance()

        if np.isnan(density).any() or np.isnan(velocity_variance).any():  
            return 0

        crowd_pressure = np.linalg.norm(density) * np.linalg.norm(velocity_variance)

        if np.isnan(crowd_pressure): 
            return 0

        return np.linalg.norm(density) , np.linalg.norm(velocity_variance) ,crowd_pressure



        self.road_network = RoadNetwork(self.data, center, radius)
        self.combined_gdf, self.intersection_dict, self.road_dict = self.road_network.process_data()
    
    # 사람수
    # def pedestrian_num(self):



# 모델 액션
    # def all_predict(self,model_weights):
        model_weights = {'1994744827':{0: '-', 8: '-', 9: 0.02, 15: 0.02}} 
        self.G, self.intersection_dict = self.road_network.predict_network(self.intersection_dict, self.road_dict,model_weights)

# 모델 상태 
    # def get_state(self):

# 모델 보상
    # def get_reward(self):

        

# 환경 리셋
    def reset(self):
        return np.array([[1,1,1] for i in range(len(list(self.intersection_dict.keys())))])


    def update(self, frame):
        if frame % self.tm == 0:  
            self.make_pedestrians()

        new_pedestrians = []
        crowd_pressure_density, crowd_pressure_velocity ,cd = self.calculate_crowd_pressure(1.0)
        self.crowd_pressure_list_density.append(crowd_pressure_density)
        self.crowd_pressure_list_velocity.append(crowd_pressure_velocity)
        self.count_pedestrains.append(len(self.pedestrians))


        if self.plot:
            # # Plot crowd pressure
            self.ax2.clear()
            self.ax2.plot(self.crowd_pressure_list_density, 'r')
            self.ax2.set_title('Crowd Pressure Over Time')
            self.ax2.set_xlabel('Time')
            self.ax2.set_ylabel('Pressure')

            self.ax3.clear()
            self.ax3.plot(self.crowd_pressure_list_velocity, 'r')
            self.ax3.set_title('Crowd Pressure Over Time')
            self.ax3.set_xlabel('Time')
            self.ax3.set_ylabel('Pressure')

            
            
            self.ax4.clear()
            self.ax4.plot(self.count_pedestrains, 'b', label='Cumulative Count Y > 90')
            self.ax4.set_title('Cumulative Count of Pedestrians Outside Y Range')
            self.ax4.set_xlabel('Time')
            self.ax4.set_ylabel('Count')
            self.ax4.legend()

        box_positions = {k: v['road_position'] for k, v in self.intersection_dict.items()}

        for p in self.pedestrians:
            # if abs(p.position[0] - self.intersection_dict[p.recommendation[0][0][-1]]['road_position'][0]) >= self.box_size or abs(p.position[1] - self.intersection_dict[p.recommendation[0][0][-1]]['road_position'][1]) >= self.box_size:
            if abs(p.position[0] - self.shop_dict[p.end_point]['location'][0]) >= self.box_size or abs(p.position[1] - self.shop_dict[p.end_point]['location'][1]) >= self.box_size:
                new_pedestrians.append(p)

            else:
                self.end_time_list.append(p.stay_time)

        self.end_time_list = self.end_time_list[-1000:]
        self.pedestrians = new_pedestrians

        all_pedestrians = np.array(self.pedestrians)
        for p in all_pedestrians:
            p.update_velocity(all_pedestrians, self.walls, 0.1)
            p.update_position(all_pedestrians, self.walls, 0.1)
            p.destination, change  = p.make_target_purpose(p.recommendation, self.intersection_dict,self.shop_dict)
            if change:
                if p.target_box_index in [0,1]:
                    target_index_box = 0
                else:
                    target_index_box = p.target_box_index-2

                p.after_destination = p.recommendation[0][0][target_index_box]

                if p.recommendation[0][0][p.target_box_index-1] not in self.intercection_point_list.keys():
                    self.intercection_point_list[p.recommendation[0][0][p.target_box_index-1]] = [[],[]]
                self.intercection_point_list[p.recommendation[0][0][p.target_box_index-1]][0].append(p.after_destination)

                if len(self.intercection_point_list[p.recommendation[0][0][p.target_box_index-1]][0]) > 100:
                    self.intercection_point_list[p.recommendation[0][0][p.target_box_index-1]][0].pop(0)

            else:
    
                if not p.recommendation[0][0][p.target_box_index-1] in self.intercection_point_list.keys():
                    self.intercection_point_list[p.recommendation[0][0][p.target_box_index-1]] = [[],[]]
#                 print(self.intercection_point_list, p.recommendation[0][0] , [p.target_box_index-1])
                self.intercection_point_list[p.recommendation[0][0][p.target_box_index-1]][0].append(0)
                if len(self.intercection_point_list[p.recommendation[0][0][p.target_box_index-1]][0]) > 100:
                    self.intercection_point_list[p.recommendation[0][0][p.target_box_index-1]][0].pop(0)

            gg = len(set(self.intercection_point_list[p.recommendation[0][0][p.target_box_index-1]][0]))
            targne = list(Counter(self.intercection_point_list[p.recommendation[0][0][p.target_box_index-1]][0]).values())
            if gg != 3:
                targne += [1] * (3 - gg)

            self.intercection_point_list[p.recommendation[0][0][p.target_box_index-1]][1] = targne

        if self.plot:
            self.scat.set_offsets([p.position for p in self.pedestrians])

        next_state = self.intercection_point_list
        if len(self.end_time_list) ==0:
            time_list_num = 1
            time_list_sum = 1
            re_rime = time_list_sum / time_list_num
        else:
            time_list_num = len(self.end_time_list)
            time_list_sum = sum(self.end_time_list)
            re_rime = time_list_sum / time_list_num
        reward = crowd_pressure_velocity*crowd_pressure_density #+ len(self.pedestrians)/500 #+ re_rime

        return next_state, -reward , crowd_pressure_density , crowd_pressure_velocity , len(self.pedestrians) , self.end_time_list

    def snap_store_to_nearest_wall(self,store, walls):
        min_distance = float('inf')
        snapped_point = None

        for wall in walls:
            distance = wall.distance_to(store)
            if distance < min_distance:
                min_distance = distance
                snapped_point = wall.closest_point_to(store)

        return snapped_point

    def play(self, tm,plot):
        self.tm = tm
        self.plot = plot
        self.box_size = 10
        box_size = 10
        self.crowd_pressure_list_density = []

        if  self.plot:
            fig = plt.figure(figsize=(10, 20))
            gs = fig.add_gridspec(4, 1, height_ratios=[2,1,1,1])
            self.ax1 = fig.add_subplot(gs[0])
            self.ax2 = fig.add_subplot(gs[1])
            self.ax3 = fig.add_subplot(gs[2])
            self.ax4 = fig.add_subplot(gs[3])

            for wall in self.walls:
                self.ax1.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]], 'k')

            for vc in self.intersection_dict:
                box_x = self.intersection_dict[vc]['road_position'][0]
                box_y = self.intersection_dict[vc]['road_position'][1]
                self.ax1.scatter(box_x,box_y,c = 'r',alpha = 0.3,s = 100*box_size)

            for place in self.shop_dict:
                mdl = self.shop_dict[place]
                new_location = self.snap_store_to_nearest_wall(np.array([mdl['location'][0], mdl['location'][1]]), self.walls)
                self.ax1.scatter(*new_location,s= 300 * mdl['rating']+500, c=(mdl['rating']/5,0,0))

            self.scat = self.ax1.scatter([p.position[0] for p in self.pedestrians], [p.position[1] for p in self.pedestrians], s=[p.size*50 for p in self.pedestrians])


            return fig
    


    def run_simulation(self,tm):
        self.make_pedestrians()
        self.make_road()
        self.play(tm,plot = False) 
        self.plot = False
        for frame in tqdm(range(1000)):
            self.update(frame) # 함수안의 함수를 들고옴 


# predict =0 train(모델로 가중치 변경) , predict =1 random (모델을 쓰지만 학습안된 모델),  predict =2 최단거리 (모델없이 가중치 길 거리로 유지하고 빠른길만)
def Trian(env,tm,plot,predict,start,set_time): 
    reward_list = []
    pressure_density_list = []
    pressure_velocity_list = []
    len_pedestrians_list = []

    fig = env.play(tm,plot)
    # 에이전트 생성
    state_size = 3  # 상태의 크기
    action_size = 3  # 가능한 행동의 수
    agent = AGENT.ActorCriticAgent(state_size, action_size)

    if not start[0]: # start[0] = False , model 불러오기  
        agent.model = start[1]

    states = env.reset()  # 환경 초기화 # 각 모퉁이의 state 집합
    target_models = []
    # 학습 루프
    for episode in tqdm(range(set_time)):  # 1000 에피소드 동안 학습
        for pp in env.pedestrians: 
            recommendations = env.navi.dijkstra(pp.start_intersection_point, pp.end_intersection_point, num_recommendations=1)
            pp.recommendation = recommendations
            pp.stay_time += 1
        dataes = []
        model_weights = []
        for state in states: 
            dataes.append([np.array(state), np.array(agent.act(np.array(state)))])  # 에이전트에 따른 행동 결정
            # model_weights['모델위치'] = np.array(agent.act(np.array(state)))
            model_weights.append(np.array(agent.act(np.array(state))))
        if predict == 0 or predict == 1:
            if not episode == 0:
                for model_weight,target_model in zip(model_weights,target_models):
                    gggg, intersection_dictlkl = env.road_network.predict_network(env.intersection_dict,model_weight,target_model)
                    env.intersection_dict = intersection_dictlkl
                    env.G = gggg
                    env.navi.graph = gggg
        target_models = []
        # if episode % 50 == 1:
        #     env.road_network.draw_graph(env.combined_gdf, env.intersection_dict, env.G)
        next_states, reward ,pressure_density ,pressure_velocity ,len_pedestrians,stay_list = env.update(episode) # 행동에 따른 다음 상태, 보상
        # for data in dataes:
        #     agent.store_transition(np.array(data[0]), np.array(data[1]), np.array([reward]))  # 에이전트가 경험을 저장하도록 함
        next_statess = []
        for i ,c in next_states.items():
            next_statess.append(c[1])
            target_models.append(i)
        try:
            states = np.array(next_statess)  # 상태 업데이트
        except:
            pass
        if predict == 0:
            agent.train()  

        
        reward_list.append(reward)
        pressure_density_list.append(pressure_density)
        pressure_velocity_list.append(pressure_velocity)
        len_pedestrians_list.append(len_pedestrians)

    return reward_list , env.navi.save_rec , pressure_density_list , pressure_velocity_list , len_pedestrians_list,agent , stay_list
