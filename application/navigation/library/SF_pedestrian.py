import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import pandas as pd
from tqdm import tqdm 


class Pedestrian:
    def __init__(self, position, destination, desired_speed, relaxation_time ,mass,size,color):
        self.position = np.array(position, dtype=float)   # 현재 위치
        self.destination = np.array(destination, dtype=float)  # 목적지
        self.desired_speed = desired_speed  # 원하는 속도
        self.relaxation_time = relaxation_time  # 완화 시간
        self.mass = mass # 보행자의 무게 
        self.size = size # 보행자의 크기
        self.color = color # 보행자의 색
        
        
        
        self.velocity = np.array([random.random(), random.random()])  # 현재 속도 (벡터)
        self.speed = np.linalg.norm(self.velocity)  # 현재 속력 (스칼라)
        self.r_alpha_beta_after = [0,0] # 나중 벡터차
        
        
        self.max_repulsive_force = 100  # 최대 상호작용 힘
        self.max_speed = 10  # 최대 속력
        
        self.find_range = 4 # 사람간의 상호작용 거리
        
        self.max_wall_force = 5 # 최대 벽과의 힘
        self.find_wall_range = 3 # 벽과의 상호작용 거리
        
        
   
        



    def desired_direction(self): 
        # 원하는 방향 e 를 계산하는 메서드 -> 논문 1번쨰 함수
        direction = self.destination - self.position
        vec_size = np.linalg.norm(direction)
        e_vec = direction / vec_size
        return e_vec
    
    def driving_force(self):
        # 내가 원하는 방향으로 갈떄 받는 제 1힘
        F = ((self.desired_speed * self.desired_direction() - self.velocity) / self.relaxation_time) 
        return F
    
    
    
    
    
    
    def repulsive_force_pedestrian(self, other, delta_t):
        # 보행자간의 반발력을 계산하는 메서드 -> 논문 3번쨰 함수
        v_beta = other.speed
        r_alpha_beta = self.position - other.position
        
        if np.linalg.norm(r_alpha_beta) < self.find_range:
        
            e_beta = other.desired_direction() # 남의 현재 속도 단위 벡터

            # 타원의 반축 b 계산 (논문의 식 (4) 참조)
            b = np.sqrt((np.linalg.norm(r_alpha_beta) + np.linalg.norm(r_alpha_beta - v_beta * delta_t * e_beta))**2 - (v_beta * delta_t)**2) / 2

            # 반발력 함수 V_alpha_beta 계산
            V_alpha_beta = 1/b # 예시로 간단한 반발력 함수 사용 1 은 b 의 평균값 되어야함 b 값이 크면 1/1000 , 이나 1/987 이나 비슷함
            
            if np.linalg.norm(V_alpha_beta) > self.max_repulsive_force:
                V_alpha_beta = self.max_repulsive_force

            # 반발력 벡터 계산
            if np.all(self.r_alpha_beta_after == 0):
                F = V_alpha_beta * (r_alpha_beta / np.linalg.norm(r_alpha_beta)) 
            else:
                F =  V_alpha_beta * (r_alpha_beta / np.linalg.norm(r_alpha_beta)) 

            self.r_alpha_beta_after = r_alpha_beta

        
            return F*20
        
        else:
            return 0
        
        
        
    def repulsive_force_wall(self, wall):
        #벽과의 거리 계산 
        distance = wall.distance_to(self.position)
        
        if distance < self.find_wall_range:

            # 벽과 나의 유닛 벡터를 계산 
            unit_vector = (self.position - wall.closest_point_to(self.position)) / distance

            force = self.max_repulsive_force / (distance**2) * unit_vector

            if np.linalg.norm(force) > self.max_wall_force:
                force = self.max_wall_force * unit_vector *10
        else:
            force = 0

        return force




    

    def total_force(self, others, walls, delta_t):
        total_force = self.driving_force() 
        for other in others:
            if other is not self:
                total_force += self.repulsive_force_pedestrian(other, delta_t)
        for wall in walls:
            total_force += self.repulsive_force_wall(wall)
        return total_force


    

    
    def update_velocity(self, others,walls, delta_t):
        F = self.total_force(others, walls,delta_t)
        a = F / self.mass
        self.velocity += a * delta_t
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.velocity * (self.max_speed /np.linalg.norm(self.velocity) )
            
            

    def update_position(self, others, walls, delta_t):
        # 업데이트된 속도를 사용하여 위치를 업데이트
        proposed_position = self.position + self.velocity * delta_t

        # 보행자 간의 충돌을 처리
        for other in others:
            if other is not self:
                distance_to_other = np.linalg.norm(proposed_position - other.position)

                # 개인 공간을 침범하는 경우 이동하지 않고 다음 보행자로 이동
                if distance_to_other < (self.size + other.size):
                    # 두 보행자 사이의 겹치는 부분을 계산
                    overlap = self.size + other.size - distance_to_other

                    # 겹치지 않도록 보행자의 위치를 조정
                    direction = (proposed_position - other.position) / distance_to_other
                    proposed_position += direction * overlap

        # 벽과의 충돌을 처리
    def update_position(self, others, walls, delta_t):
        previous_position = self.position
        proposed_position = self.position + self.velocity * delta_t

        # 보행자 간의 충돌을 처리
        for other in others:
            if other is not self:
                distance_to_other = np.linalg.norm(proposed_position - other.position)

                # 개인 공간을 침범하는 경우 이동하지 않고 다음 보행자로 이동
                if distance_to_other < (self.size + other.size):
                    # 두 보행자 사이의 겹치는 부분을 계산
                    overlap = self.size + other.size - distance_to_other

                    # 겹치지 않도록 보행자의 위치를 조정
                    direction = (proposed_position - other.position) / distance_to_other
                    proposed_position += direction * overlap

        for wall in walls:
            if wall.distance_to(proposed_position) < self.size:
                closest_point_on_wall = wall.closest_point_to(proposed_position)

                proposed_position = closest_point_on_wall + (self.size * (self.position - closest_point_on_wall) / wall.distance_to(self.position))

                movement_vector = self.position - proposed_position

                wall_vector = wall.end - wall.start
                wall_vector /= np.linalg.norm(wall_vector)

                perpendicular_component = movement_vector - np.dot(movement_vector, wall_vector) * wall_vector

                self.velocity = perpendicular_component / delta_t

                self.position = previous_position
                self.position += self.velocity * delta_t

        self.position = proposed_position

    def in_target_box(self, target_box):
        x, y = self.position
        box_x, box_y = target_box[0]
        box_size = target_box[1]
        if abs(x - box_x) <= box_size and abs(y - box_y) <= box_size:
            return True
        return False
