import numpy as np


class Wall:
    def __init__(self, start, end):
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)

    def distance_to(self, point):
        direction = self.end - self.start
        magnitude = np.linalg.norm(direction)
        if magnitude == 0:
            return np.linalg.norm(self.start - point)

        # 벽의 방향 벡터를 단위 벡터로 정규화함
        unit_direction = direction / magnitude

        # 벽의 시작점에서 주어진 점까지의 벡터
        point_vector = point - self.start

        # 벡터의 투영 길이를 계산하여 벽의 선분을 벗어난지 여부 확인하기
        projection_length = np.dot(point_vector, unit_direction)
        if projection_length <= 0:
            # 벽의 시작점보다 왼쪽에 위치한 경우
            return np.linalg.norm(self.start - point)
        elif projection_length >= magnitude:
            # 벽의 끝점보다 오른쪽에 위치한 경우
            return np.linalg.norm(self.end - point)

        # 벽의 선분 상에서의 가장 가까운 거리 계산
        closest_distance = np.abs(np.cross(point_vector, unit_direction))
        return closest_distance

    def closest_point_to(self, point):
        direction = self.end - self.start
        magnitude = np.linalg.norm(direction)
        if magnitude == 0:
            return self.start
        t = np.dot(point - self.start, direction) / magnitude**2
        t = np.clip(t, 0, 1)  
        closest_point = self.start + t * direction
        return closest_point


