from SF_pedestrian import Pedestrian
from SF_wall import Wall
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm 
from matplotlib.patches import Rectangle


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
        self.f = []
        self.count_list = []
        self.count_targets_all = []
        self.count_pedestrains = []
        self.count_targets = [[]] * len(self.target)
        self.crowd_pressure_list_density = []
        self.crowd_pressure_list_velocity = []
        self.crowd_pressure_x = []
        self.width = width
        self.length = length

        self.pedestrians_in_store = {}  # key: 보행자, value: 상점에 들어간 프레임


    def calculate_density(self, R):
        if len(self.pedestrians) == 0:
            return np.array([0])  

        density = np.zeros(len(self.pedestrians))
        for i, p_i in enumerate(self.pedestrians):
            for j, p_j in enumerate(self.pedestrians):
                if i != j:
                    distance = np.linalg.norm(p_i.position - p_j.position)
                    local_density = np.exp(-(distance)**2 / (R**2)) 
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

     
            
    def make_pedestrians(self, type_):
        for _ in range(self.num_pedestrians):
            target_index = random.choice(range(len(self.target)))
            color_ = random.choice(['r', 'b']) if type_ == 1 else self.color_set[target_index]

            last_30_positions = [p.position for p in self.pedestrians[-10:]]
            position_range = (-self.width + 2, self.width - 2)

            for _ in range(100):
                if type_ == 1:
                    if color_ == 'r':
                        position = np.array([random.uniform(*position_range),self.length-10])
                    else:
                        position = np.array([random.uniform(*position_range),-self.length+10])
                else:
                    position = np.array([random.uniform(*position_range),random.choice([self.length-10,-self.length+10])])
                if all(np.linalg.norm(position - pos) > 3 for pos in last_30_positions):
                    break

            target = np.array([random.uniform(*position_range), -self.length if color_ == 'r' else self.length]) if type_ == 1 \
                    else self.target[target_index][0] + np.array([self.target[target_index][0][0]*0.1, 0])


    
            pedestrian = Pedestrian(1.1 * position, 1.1 * target, 5 + 0.5 * random.random(), 0.5, 1, 1, color_)
            pedestrian.target = target_index


            self.pedestrians.append(pedestrian)
            self.colors.append(color_)


    def make_store(self, walls, targets):
        # 가게의 위치에 따라 왼쪽 또는 오른쪽에 가게 생성
        direction = 'left' if targets[0][0] < 0 else 'right'
        
        # 벽 리스트 주면 가게 추가
        size_d = 5
        size = targets[1]
        x, y = targets[0][0], targets[0][1]

        if direction == 'left':
            walls.append(Wall([x, y-size],[x-size_d*3, y-size]))
            walls.append(Wall([x, y+size],[x-size_d*3, y+size]))
            walls.append(Wall([x-size_d*3, y-size],[x-size_d*3, y+size]))
        else:
            walls.append(Wall([x, y-size],[x+size_d*3, y-size]))
            walls.append(Wall([x, y+size],[x+size_d*3, y+size]))
            walls.append(Wall([x+size_d*3, y-size],[x+size_d*3, y+size]))
        
        # 가게 주변에 벽 추가
        walls += [Wall([x, y - size], [x, -self.length]),Wall([x, y + size], [x, self.length])]

        # 반대편에는 벽만 추가
        walls.append(Wall([-x, -self.length], [-x, self.length]))

        return walls

    def make_road(self, type_):
        if not type_ == 1:
            # 가게들을 왼쪽 가게와 오른쪽 가게로 분리
            left_targets = sorted([target for target in self.target if target[0][0] < 0], key=lambda x: x[0][1])
            right_targets = sorted([target for target in self.target if target[0][0] > 0], key=lambda x: x[0][1])

            for targets in [left_targets, right_targets]:

                # 가게를 벽 리스트에 추가
                for target in targets:
                    self.walls += self.make_store(self.walls, target)
        else:
            self.walls += [Wall([x, self.length], [x, -self.length]) for x in [-self.width, self.width]]




def start(env,type_,tm,plot):
    tracker = TimeSeriesTracker(patience=100)
    fig = plt.figure(figsize=(13,45))
    env.make_road(type_)
    env.make_pedestrians(type_)
    if plot:


        gs = fig.add_gridspec(6, 1, height_ratios=[2, 1, 1, 1, 1,1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax4 = fig.add_subplot(gs[3])
        ax5 = fig.add_subplot(gs[4])
        ax6 = fig.add_subplot(gs[5])


        
        for wall in env.walls:
            ax1.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]], 'k')

        
        if not type_ == 1:
            for target,i in zip(env.target,range(len(env.target))):
                color = env.color_set[i]  
                ax1.add_patch(Rectangle((target[0][0] - 2, target[0][1] - target[1]), 4, 2 * target[1], edgecolor=color, facecolor=color, alpha=0.5))


        ax1.set_xlim(-50, 50)
        ax1.set_ylim(-env.length-20, env.length+20)
        scat = ax1.scatter([p.position[0] for p in env.pedestrians], [p.position[1] for p in env.pedestrians], s=[p.size*50 for p in env.pedestrians], c=env.colors)

    crowd_pressure_list_density = []
    crowd_pressure_list_velocity = []
    crowd_pressure_x = []

    def update_1(frame):
        if frame % tm == 0:  
            env.make_pedestrians(type_)
        count_red = sum(1 for p in env.pedestrians if p.position[1] > env.length - 5)
        count_blue = sum(1 for p in env.pedestrians if p.position[1] < -env.length +5)
        env.pedestrians = [p for p in env.pedestrians if p.position[1] > -env.length +5 and p.position[1] < env.length -5]
        env.colors = [p.color for p in env.pedestrians]
        
        for p in env.pedestrians:
            env.walls = [Wall([-env.width, -env.length], [-env.width, env.length]), Wall([env.width, -env.length], [env.width, env.length])]
            p.update_velocity(env.pedestrians, env.walls, 0.1)
            p.update_position(env.pedestrians, env.walls, 0.1)
        if plot:
            try:
                scat.set_offsets([p.position for p in env.pedestrians])
            except:
                pass
            scat.set_color(env.colors)
            

        env.count_list.append([count_red, count_blue])
        env.count_pedestrains.append(len(env.pedestrians))
        cumulative_count = np.cumsum(env.count_list, axis=0)
        
        crowd_pressure_density,crowd_pressure_velocity ,cd = env.calculate_crowd_pressure(1.0)
        env.crowd_pressure_list_density.append(crowd_pressure_density)
        env.crowd_pressure_list_velocity.append(crowd_pressure_velocity)
        env.crowd_pressure_x.append(cd)


        if plot:
            ax2.clear()
            ax2.plot(env.crowd_pressure_list_density, 'r')
            ax2.set_title('Crowd Density Over Time')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Density')

            ax3.clear()
            ax3.plot(env.crowd_pressure_list_velocity, 'b')
            ax3.set_title('Crowd Velocity Over Time')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Velocity')
            
            ax4.clear()
            ax4.plot(env.crowd_pressure_x, 'black')
            ax4.set_title('Crowd Pressure Over Time')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Pressure')
            
            ax5.clear()
            ax5.plot(env.count_pedestrains, 'b', label='Cumulative Count inside')
            ax5.set_title('Cumulative Count of Pedestrians Inside')
            ax5.set_xlabel('Time')
            ax5.set_ylabel('Count')
            ax5.legend()


            ax6.clear()
            ax6.plot(cumulative_count[:, 0], 'b', label='Cumulative Count up outside')
            ax6.plot(cumulative_count[:, 1], 'r', label='Cumulative Count down outside')
            ax6.set_title('Cumulative Count of Pedestrians Outside')
            ax6.set_xlabel('Time')
            ax6.set_ylabel('Count')
            ax6.legend()


            



        if not plot:
            plt.close()
        tracker.update(len(env.pedestrians))
        if tracker.should_stop(): # 현상태 유지 
            try:
                # ani.event_source.stop()
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,0
            except:
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,0


        elif len(env.count_pedestrains) == 1000: # 압사수 계속 증가
            try:
                # ani.event_source.stop()
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,1
            except:
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,1

        elif len(env.pedestrians) > 500: 
            try:
                # ani.event_source.stop()
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,2
            except:
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,2


        else:
            return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,True


    def update_2(frame):
        if frame % tm == 0:
            env.make_pedestrians(type_)
            
        env.pedestrians = [p for p in env.pedestrians if p.position[0] > -env.width -1 and p.position[0] < env.width +1]
        env.colors = [p.color for p in env.pedestrians]
        
        for p in env.pedestrians:
            env.walls = []
            targets = env.target[p.target] if p.target < len(env.target) else env.target[-1]
            env.walls = env.make_store(env.walls, targets)
            p.update_velocity(env.pedestrians, env.walls, 0.1)
            p.update_position(env.pedestrians, env.walls, 0.1)
        if plot:
            try:
                scat.set_offsets([p.position for p in env.pedestrians]) 
            except:
                pass

            scat.set_color(env.colors)
        
        count_targets_box = [0] * len(env.target)
        for p in env.pedestrians:
            if p.target < len(env.target) and p.in_target_box(env.target[p.target]) and p.position[0] <= -env.width -1 or p.position[0] >= env.width +1:
                count_targets_box[p.target] += 1
        
        env.count_targets_all.append(count_targets_box.copy())
        env.count_pedestrains.append(len(env.pedestrians))
        

        
        crowd_pressure_density,crowd_pressure_velocity,cc  = env.calculate_crowd_pressure(1.0)
        env.crowd_pressure_list_density.append(crowd_pressure_density)
        env.crowd_pressure_list_velocity.append(crowd_pressure_velocity)
        env.crowd_pressure_x.append(cc)


        if plot:
            ax2.clear()
            ax2.plot(env.crowd_pressure_list_density, 'r')
            ax2.set_title('Crowd Density Over Time')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Density')

            ax3.clear()
            ax3.plot(env.crowd_pressure_list_velocity, 'b')
            ax3.set_title('Crowd Velocity Over Time')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Velocity')
            
            ax4.clear()
            ax4.plot(env.crowd_pressure_x, 'black')
            ax4.set_title('Crowd Pressure Over Time')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Pressure')
            
            ax5.clear()
            ax5.plot(env.count_pedestrains, 'b', label='Cumulative Count inside')
            ax5.set_title('Cumulative Count of Pedestrians Inside')
            ax5.set_xlabel('Time')
            ax5.set_ylabel('Count')
            ax5.legend()

            ax6.clear()
            colors = env.color_set[:len(env.target)]
            for i, target in enumerate(env.target):
                ax6.plot(np.cumsum([count[i] for count in env.count_targets_all]), color=colors[i], label='Target {}'.format(i+1))
            ax6.set_title('Cumulative Count of Pedestrians Inside Target Boxes')
            ax6.set_xlabel('Time')
            ax6.set_ylabel('Count')
            ax6.legend()



        if not plot:
            plt.close()
        tracker.update(len(env.pedestrians))
        if tracker.should_stop(): # 현상태 유지 
            try:
                # ani.event_source.stop()
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,0
            except:
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,0


        elif len(env.count_pedestrains) == 1000: # 압사수 계속 증가
            try:
                # ani.event_source.stop()
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,1
            except:
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,1

        elif len(env.pedestrians) > 500: # 압사 확실 
            try:
                # ani.event_source.stop()
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,2
            except:
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,2


        else:
            return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,True

    def update_3(frame):
        if frame % tm == 0:  
            env.make_pedestrians(type_)
            

        env.pedestrians = [p for p in env.pedestrians if p.position[1] > -env.length +5 and p.position[1] < env.length -5]
        env.pedestrians = [p for p in env.pedestrians if p.position[0] > -env.width -1 and p.position[0] < env.width +1]
        env.colors = [p.color for p in env.pedestrians]
        
        for p in env.pedestrians:
            env.walls = []
            targets = env.target[p.target] if p.target < len(env.target) else env.target[-1]
            env.walls = env.make_store(env.walls, targets)
            p.update_velocity(env.pedestrians, env.walls, 0.1)
            p.update_position(env.pedestrians, env.walls, 0.1)
        if plot:
            try:
                scat.set_offsets([p.position for p in env.pedestrians]) 
            except:
                pass

            scat.set_color(env.colors)
        
        count_targets_box = [0] * len(env.target)
        for p in env.pedestrians:
            if p.target < len(env.target) and p.in_target_box(env.target[p.target]) and p.position[0] <= -env.width -1 or p.position[0] >= env.width +1:
                env.count_targets[p.target].append(0)
                count_targets_box[p.target] += 1
        
        env.count_targets_all.append(count_targets_box.copy())
        env.count_pedestrains.append(len(env.pedestrians))


        env.count_targets = [[x+1 for x in sub_list] for sub_list in env.count_targets]

        removed_indexes = []
        for i, sub_list in enumerate(env.count_targets): # 생성해할 사람 알려주는 
            for j in reversed(range(len(sub_list))):  
                if sub_list[j] > 5:
                    del sub_list[j]
                    removed_indexes.append(i)


        for make_target in removed_indexes:
            target_index = make_target
            color_ = env.color_set[target_index]

            last_30_positions = [p.position for p in env.pedestrians[-10:]]
            position_range = (-env.width + 2, env.width - 2)


            target = np.array([random.uniform(*position_range),random.choice([env.length+10,-env.length-10])])


            position = env.target[target_index][0] - np.array([env.target[target_index][0][0]*0.3, 0])
        


    
            pedestrian = Pedestrian(1.1 * position, 1.1 * target, 5 + 0.5 * random.random(), 0.5, 1, 1, color_)
            pedestrian.target = target_index


            env.pedestrians.append(pedestrian)
            env.colors.append(color_)


        

        
        crowd_pressure_density,crowd_pressure_velocity,cc  = env.calculate_crowd_pressure(1.0)
        env.crowd_pressure_list_density.append(crowd_pressure_density)
        env.crowd_pressure_list_velocity.append(crowd_pressure_velocity)
        env.crowd_pressure_x.append(cc)


        if plot:
            ax2.clear()
            ax2.plot(env.crowd_pressure_list_density, 'r')
            ax2.set_title('Crowd Density Over Time')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Density')

            ax3.clear()
            ax3.plot(env.crowd_pressure_list_velocity, 'b')
            ax3.set_title('Crowd Velocity Over Time')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Velocity')
            
            ax4.clear()
            ax4.plot(env.crowd_pressure_x, 'black')
            ax4.set_title('Crowd Pressure Over Time')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Pressure')
            
            ax5.clear()
            ax5.plot(env.count_pedestrains, 'b', label='Cumulative Count inside')
            ax5.set_title('Cumulative Count of Pedestrians Inside')
            ax5.set_xlabel('Time')
            ax5.set_ylabel('Count')
            ax5.legend()

            ax6.clear()
            colors = env.color_set[:len(env.target)]
            for i, target in enumerate(env.target):
                ax6.plot(np.cumsum([count[i] for count in env.count_targets_all]), color=colors[i], label='Target {}'.format(i+1))
            ax6.set_title('Cumulative Count of Pedestrians Inside Target Boxes')
            ax6.set_xlabel('Time')
            ax6.set_ylabel('Count')
            ax6.legend()


        if not plot:
            plt.close()
        tracker.update(len(env.pedestrians))
        if tracker.should_stop(): # 현상태 유지 
            try:
                # ani.event_source.stop()
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,0
            except:
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,0


        elif len(env.count_pedestrains) == 1000: # 압사수 계속 증가
            try:
                # ani.event_source.stop()
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,1
            except:
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,1

        elif len(env.pedestrians) > 500: 
            try:
                # ani.event_source.stop()
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,2
            except:
                return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,2


        else:
            return env.count_pedestrains[-1],crowd_pressure_density,crowd_pressure_velocity,True



    if type_ == 1:
        
        return fig, update_1
    
    elif type_ == 2:
        
        return fig, update_2

    elif type_ == 3:
        
        return fig, update_3

def run_simulation(type_ , target, width, length , plot):
    count_stepss = 300
    crowed_time_list_in = []
    crowed_time_list = []

    crowd_pressure_density_list = []
    crowd_pressure_density_list_in = []

    crowd_pressure_velocity_list = []
    crowd_pressure_velocity_list_in = []

    crowed_pedestrian_count_list = []
    crowed_pedestrian_count_list_in = []

    for step in tqdm(range(count_stepss)):
        for i in range(10):
            tm = i+2
            num = 1
            env1 = ENV(num,target,width,length)

            env1.make_pedestrians(type_)

            fig, update_m = start(env1, type_, tm,plot)

            stc = 0
            for frame in tqdm(range(1000)):
                crowd_count_pedestrains,crowd_pressure_density,crowd_pressure_velocity,return_ = update_m(frame) 
                break

            crowed_pedestrian_count_list_in.append(crowd_count_pedestrains)
            crowed_time_list_in.append(frame)
            crowd_pressure_density_list_in.append(crowd_pressure_density)
            crowd_pressure_velocity_list_in.append(crowd_pressure_velocity)
        crowed_pedestrian_count_list.append(crowed_pedestrian_count_list_in)
        crowed_time_list.append(crowed_time_list_in)
        crowd_pressure_density_list.append(crowd_pressure_density_list_in)
        crowd_pressure_velocity_list.append(crowd_pressure_velocity_list_in)

        crowed_pedestrian_count_list_in = []
        crowed_time_list_in = []
        crowd_pressure_density_list_in = []
        crowd_pressure_velocity_list_in = []

    return crowed_pedestrian_count_list, crowed_time_list , crowd_pressure_density_list , crowd_pressure_velocity_list

def plot_graph(data_list, title):
    count_stepss = 300
    plt.figure(figsize=(20,20))
    for data in data_list:
        plt.plot(data)
    plt.title(title)
    plt.show()

    total = np.sum(data_list, axis=0)/count_stepss
    plt.figure(figsize=(20,20))
    plt.plot(total)
    plt.title(title + ' - Total')
    plt.show()

def plot_steps(crowed_count_list , crowed_time_list , crowd_pressure_density_list , crowd_pressure_velocity_list):
    plot_graph(crowed_count_list, 'crowed_count')
    plot_graph(crowed_time_list, 'Steps')
    plot_graph(crowd_pressure_density_list, 'Crowd Pressure Density')
    plot_graph(crowd_pressure_velocity_list, 'Crowd Pressure Velocity')
    
    
