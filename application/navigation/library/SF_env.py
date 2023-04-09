from navigation.library.SF_pedestrian import Pedestrian
from navigation.library.SF_wall import Wall
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import pandas as pd
from tqdm import tqdm 
from matplotlib.patches import Rectangle
from scipy.ndimage.filters import gaussian_filter1d


def generate_targets(type_,width,length):
    def is_overlap(new_target, existing_targets):
        for existing_target in existing_targets:
            if new_target[0][0] != existing_target[0][0]:  
                continue
            if abs(new_target[0][1] - existing_target[0][1]) < (new_target[1] + existing_target[1]):
                return True  
        return False

    target = []
    if type_ == 1:
        for _ in range(2):
            while True:
                new_target = [[random.choices([-width, width])[0], random.choice([-length +20, length -20])], random.randint(5, 7)]
                if not is_overlap(new_target, target):
                    target.append(new_target)
                    break
    else:
        for _ in range(7):
            while True:
                new_target = [[random.choices([-width, width])[0], random.randint(-length +20, length -20)], random.randint(7, 10)]
                if not is_overlap(new_target, target):
                    target.append(new_target)
                    break
    return target

class TimeSeriesTracker:
    def __init__(self, patience=100):
        self.patience = patience
        self.best_score = float('-inf')
        self.steps_since_best = 0

    def update(self, current_score):
        if current_score > self.best_score:
            self.best_score = current_score
            self.steps_since_best = 0
        else:
            self.steps_since_best += 1

    def should_stop(self):
        return self.steps_since_best == self.patience





class ENV:
    def __init__(self, num_pedestrians,target,width,length): # target = [[[2,3],3],[[2,1],7]].....
        self.num_pedestrians = num_pedestrians
        self.pedestrians = []
        self.target = target
        self.walls = []
        self.color_set = ['g','r','black','yellow','c','gray','b']
        self.colors = []
        self.y_data = []
        self.f = []
        self.count_list = []
        self.count_targets_all = []
        self.count_pedestrains = []
        self.width = width
        self.length = length

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

        density[np.isnan(density)] = 0  # Replace any NaN values with 0

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
        
    def make_pedestrians(self, type_):
        for _ in range(self.num_pedestrians):
            target_index = random.choice(range(len(self.target)))
            color_ = random.choice(['r', 'b']) if type_ == 1 else self.color_set[target_index]

            # 최근 30명의 보행자 위치를 가져옵니다.
            last_30_positions = [p.position for p in self.pedestrians[-10:]]

            # 각 보행자의 위치를 랜덤하게 결정하되, 최근 30명의 보행자와는 거리가 3 이상 떨어져 있어야 합니다.

            # 보행자의 목표 위치를 결정합니다.
            target = None
            position = None

            if type_ == 1:
                if color_ =='r':
                    target = np.array([random.uniform(-self.width + 2, self.width - 2),-self.length])
                    for _ in range(100):
                        position = np.array([random.uniform(-self.width + 2, self.width - 2), self.length-10])
                        if all(np.linalg.norm(position - pos) > 3 for pos in last_30_positions):
                            break
                else:
                    target = np.array([random.uniform(-self.width + 2, self.width - 2), self.length])
                    for _ in range(100):
                        position = np.array([random.uniform(-self.width + 2, self.width - 2), -self.length+10])
                        if all(np.linalg.norm(position - pos) > 3 for pos in last_30_positions):
                            break
            elif type_ == 2:
                target = self.target[target_index][0] + np.array([self.target[target_index][0][0]*0.1, 0])
                for _ in range(100):
                    position = np.array([random.uniform(-self.width + 2, self.width - 2), random.choice([-self.length+10, self.length-10])])
                    if all(np.linalg.norm(position - pos) > 3 for pos in last_30_positions):
                        break

            # 보행자 객체를 생성하고 목록에 추가합니다.
            pedestrian = Pedestrian(1.1 * position, 1.1 * target, 5 + 0.5 * random.random(), 0.5, 1, 1, self.color_set[target_index])
            pedestrian.target = target_index
            pedestrian.color = color_
            self.pedestrians.append(pedestrian)

            self.colors.append(color_)
            self.y_data.append([self.colors.count('r'), self.colors.count('b'), len(self.pedestrians), self.calculate_crowd_pressure(4.0)])
 
    def make_store(self,walls,target,size,rl): # 벽 리스트 주면 가게 추가해주는 역할
        size_d = 5
        if rl == 'left':
            walls.append(Wall([target[0],target[1]-size],[target[0]-size_d*3,target[1]-size]))
            walls.append(Wall([target[0],target[1]+size],[target[0]-size_d*3,target[1]+size]))
            walls.append(Wall([target[0]-size_d*3,target[1]-size],[target[0]-size_d*3,target[1]+size]))

        else:
            walls.append(Wall([target[0],target[1]-size],[target[0]+size_d*3,target[1]-size]))
            walls.append(Wall([target[0],target[1]+size],[target[0]+size_d*3,target[1]+size]))
            walls.append(Wall([target[0]+size_d*3,target[1]-size],[target[0]+size_d*3,target[1]+size]))

        return walls

    def make_road(self, type_):
        walls = self.walls

        if type_ == 2:
            # 가게들을 왼쪽 가게와 오른쪽 가게로 분리
            left_targets = sorted([target for target in self.target if target[0][0] < 0], key=lambda x: x[0][1])
            right_targets = sorted([target for target in self.target if target[0][0] > 0], key=lambda x: x[0][1])

            for targets in [left_targets, right_targets]:
                direction = 'left' if targets[0][0][0] < 0 else 'right'

                # 가게를 벽 리스트에 추가
                for target in targets:
                    self.make_store(walls, target[0], target[1], direction)

                # 길의 양 끝과 가게들 사이에 벽을 추가합니다.
                for i in range(len(targets)-1):
                    x = targets[i][0][0]
                    y1 = targets[i][0][1] + targets[i][1]
                    y2 = targets[i+1][0][1] - targets[i+1][1]
                    if y1 < y2:  # 가게들 사이에 공간이 있을 경우에만 벽 추가
                        walls.append(Wall([x, y1], [x, y2]))

                # 길의 양 끝에 벽 추가
                x = targets[0][0][0]
                bottom_store = targets[0]
                top_store = targets[-1]
                walls.append(Wall([x, bottom_store[0][1] - bottom_store[1]], [x, -self.length]))
                walls.append(Wall([x, top_store[0][1] + top_store[1]], [x, self.length]))
        else:
            walls += [Wall([x, self.length], [x, -self.length]) for x in [-self.width, self.width]]

        return walls

    def make_road_one(self, walls, targets):
        # 가게의 위치에 따라 왼쪽 또는 오른쪽에 가게 생성
        direction = 'left' if targets[0][0] < 0 else 'right'
        self.make_store(walls, targets[0], targets[1], direction)

        # 가게 주변에 벽 추가
        x = targets[0][0]
        y = targets[0][1]
        size = targets[1]
        walls += [
            Wall([x, y - size], [x, -self.length]),
            Wall([x, y + size], [x, self.length])
        ]

        # 반대편에는 벽만 추가
        walls.append(Wall([-x, -self.length], [-x, self.length]))

        return walls





def moving_avg_and_smoothed_gradient(data, window_size, sigma):
    # 이동평균 계산
    weights = np.ones(window_size) / window_size
    moving_avg = np.convolve(data, weights, mode='valid')
    
    # Gaussian smoothing 적용
    smoothed_data = gaussian_filter1d(moving_avg, sigma)
    
    # 기울기 계산
    gradient = np.gradient(smoothed_data)
    
    return moving_avg, gradient

def start(env,type_,tm):
    tracker = TimeSeriesTracker(patience=100)

    fig = plt.figure(figsize=(10, 40))

    gs = fig.add_gridspec(5, 1, height_ratios=[2, 1, 1, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4])


    for wall in env.walls:
        ax1.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]], 'k')

    
    for target,i in zip(env.target,range(len(env.target))):
        color = env.color_set[i]  
        ax1.add_patch(Rectangle((target[0][0] - 2, target[0][1] - target[1]), 4, 2 * target[1], edgecolor=color, facecolor=color, alpha=0.5))


    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-env.length-50, env.length+50)
    scat = ax1.scatter([p.position[0] for p in env.pedestrians], [p.position[1] for p in env.pedestrians], s=[p.size*50 for p in env.pedestrians], c=env.colors)

    crowd_pressure_list_density = []
    crowd_pressure_list_velocity = []
    crowd_pressure_x = []

    def update_1(frame):
        if frame % tm == 0:  
            env.make_pedestrians(type_)
        print([p.position[1] for p in env.pedestrians])
        env.pedestrians = [p for p in env.pedestrians if p.position[1] > -env.length +5 and p.position[1] < env.length -5]
        print(len(env.pedestrians))
        env.colors = [p.color for p in env.pedestrians]
        
        for p in env.pedestrians:
            env.walls = [Wall([-env.width, -env.length], [-env.width, env.length]), Wall([env.width, -env.length], [env.width, env.length])]
            p.update_velocity(env.pedestrians, env.walls, 0.1)
            p.update_position(env.pedestrians, env.walls, 0.1)
        scat.set_offsets([p.position for p in env.pedestrians])
        scat.set_color(env.colors)
        
        # Calculate count of pedestrians outside Y range
        count_red = sum(1 for p in env.pedestrians if p.position[1] > env.length - 10)
        count_blue = sum(1 for p in env.pedestrians if p.position[1] < -env.length +10)
        env.count_list.append([count_red, count_blue])
        env.count_pedestrains.append(len(env.pedestrians))
        cumulative_count = np.cumsum(env.count_list, axis=0)
        
        # Plot cumulative count of pedestrians outside Y range
        ax4.clear()
        ax4.plot(cumulative_count[:, 0], 'b', label='Cumulative Count Y > 90')
        ax4.plot(cumulative_count[:, 1], 'r', label='Cumulative Count Y < -90')
        ax4.set_title('Cumulative Count of Pedestrians Outside Y Range')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Count')
        ax4.legend()

        ax5.clear()
        ax5.plot(env.count_pedestrains, 'b', label='Cumulative Count Y > 90')
        ax5.set_title('Cumulative Count of Pedestrians Outside Y Range')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Count')
        ax5.legend()
        
        # Calculate crowd pressure and add to list
        crowd_pressure_density,crowd_pressure_velocity ,cd = env.calculate_crowd_pressure(1.0)
        crowd_pressure_list_density.append(crowd_pressure_density)
        crowd_pressure_list_velocity.append(crowd_pressure_velocity)
        crowd_pressure_x.append(cd)

        # Calculate moving average and gradient
        # if len(crowd_pressure_list) > 31:
        #     moving_avg, gradient = moving_avg_and_smoothed_gradient(crowd_pressure_list, 30, 30)
        # else:
        #     gradient = []

        # # Plot crowd pressure
        ax2.clear()
        ax2.plot(crowd_pressure_list_density, 'r')
        ax2.set_title('Crowd Pressure Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Pressure')

        # Plot gradient of crowd pressure
        ax3.clear()
        ax3.plot(crowd_pressure_list_velocity, 'b')
        ax3.set_title('Gradient of Crowd Pressure Over Time')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Gradient')


        tracker.update(len(env.pedestrians))
        if tracker.should_stop(): # 현상태 유지 
            try:
                ani.event_source.stop()
                return 0
            except:
                return 0


        elif len(env.count_pedestrains) == 1000: # 압사수 계속 증가
            try:
                ani.event_source.stop()
                return 1
            except:
                return 1

        elif len(env.pedestrians) > 500: # 압사 확실  -> width , length 에 맞춰서 크기 조정 필요
            try:
                ani.event_source.stop()
                return 2
            except:
                return 2


        else:
            return True


    def update_2(frame):
        if frame % tm == 0:  
            env.make_pedestrians(type_)
            
        env.pedestrians = [p for p in env.pedestrians if p.position[0] > -env.width -1 and p.position[0] < env.width +1]
        env.colors = [p.color for p in env.pedestrians]
        
        for p in env.pedestrians:
            env.walls = []
            gg = env.target[p.target] if p.target < len(env.target) else env.target[-1]
            env.walls = env.make_road_one(env.walls, gg)
            p.update_velocity(env.pedestrians, env.walls, 0.1)
            p.update_position(env.pedestrians, env.walls, 0.1)
        scat.set_offsets([p.position for p in env.pedestrians])
        scat.set_color(env.colors)
        
        count_targets = [0] * len(env.target)
        for p in env.pedestrians:
            if p.target < len(env.target) and p.in_target_box(env.target[p.target]):
                count_targets[p.target] += 1
        
        env.count_targets_all.append(count_targets.copy())
        env.count_pedestrains.append(len(env.pedestrians))
        
        ax4.clear()
        colors = env.color_set[:len(env.target)]
        for i, target in enumerate(env.target):
            ax4.plot(np.cumsum([count[i] for count in env.count_targets_all]), color=colors[i], label='Target {}'.format(i+1))
        ax4.set_title('Cumulative Count of Pedestrians Inside Target Boxes')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Count')
        ax4.legend()


        ax5.clear()
        ax5.plot(env.count_pedestrains, 'b', label='Cumulative Count Y > 90')
        ax5.set_title('Cumulative Count of Pedestrians Outside Y Range')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Count')
        ax5.legend()

        
        crowd_pressure_density,crowd_pressure_velocity,cc  = env.calculate_crowd_pressure(1.0)
        crowd_pressure_list_density.append(crowd_pressure_density)
        crowd_pressure_list_velocity.append(crowd_pressure_velocity)
        crowd_pressure_x.append(cc)

        # # Calculate moving average and gradient
        # if len(crowd_pressure_list) > 31:
        #     moving_avg, gradient = moving_avg_and_smoothed_gradient(crowd_pressure_list, 30, 30)
        # else:
        #     gradient = []

        ax2.clear()
        ax2.plot(crowd_pressure_list_density, 'r')
        ax2.set_title('Crowd Pressure Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Pressure')

        ax3.clear()
        ax3.plot(crowd_pressure_list_velocity, 'b')
        ax3.set_title('Gradient of Crowd Pressure Over Time')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Gradient')


        tracker.update(len(env.pedestrians))
        if tracker.should_stop(): # 현상태 유지 
            try:
                ani.event_source.stop()
                return 0
            except:
                return 0


        elif len(env.count_pedestrains) == 1000: # 압사수 계속 증가
            try:
                ani.event_source.stop()
                return 1
            except:
                return 1

        elif len(env.pedestrians) > 500: # 압사 확실 
            try:
                ani.event_source.stop()
                return 2
            except:
                return 2


        else:
            return True


            
        
    if type_ == 1:
        
        return fig, update_1
    
    elif type_ == 2:
        
        return fig, update_2





