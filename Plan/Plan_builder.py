# Code utiliser pour generer les archives .csv des tables "rooms" et "points" pour la base de donnees de la planification

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon


class Plan_builder:
    def __init__(self, rooms_db, points_db=None):
        self.gap = 1.3
        self.rooms_db = rooms_db
        self.points_db = points_db
        self.door_pos_dict = {"LC410":"right", "LC412":"left","LC414":"right", "LC416":"left", "LC418":"right","LC411":"right","LC413":"left","LC415":"right","LC417":"right","LC419":"left"}
    

    def _get_polygon(self, refx, refy, lx, ly):
        return [(refx, refy), (refx + lx, refy), (refx + lx, refy + ly), (refx, refy + ly), (refx, refy)]


    # door = "left" or "right" 
    # parity = "odd" or "even"
    def _get_position(self, room_part, parity, door, refx, refy, lx, ly):
        if room_part < 5:
            if parity == "even":
                if door == "left":
                    room_part = room_part%4
                else:
                    room_part = (room_part+1)%4
            if parity == "odd":
                if door == "left":
                    room_part = (room_part+3)%4
                else:
                    room_part = (room_part+2)%4  
        if room_part == 1:
            return (refx + self.gap, refy - self.gap)
        if room_part == 2:
            return (refx + lx - self.gap, refy - self.gap)
        if room_part == 3:
            return (refx + lx - self.gap, refy + ly + self.gap)
        if room_part == 0: # id == 4
            return (refx + self.gap, refy + ly + self.gap)
        if room_part == 5:
            if door == "left":
                return (refx + 0.5, 1.05)
            else:
                return (refx + lx - 0.5, 1.05)       
        
    def _get_center(self, refx, refy, lx, ly):
        return (refx + lx/2, refy + ly/2)

    def plot_walls(self):
        for index, row in self.rooms_db.iterrows():
            poly = self._get_polygon(row['refx'], row["refy"], row["lx"], row["ly"])
            p = plt.Polygon(poly, edgecolor='black',facecolor='none')
            plt.gca().add_patch(p)
            plt.gca().set_aspect('equal', adjustable='box')
        
    def plot_names(self):
        for index, row in self.rooms_db.iterrows():
            if not row["room"]=="extra_wall":
                center  = self._get_center(row["refx"], row["refy"], row["lx"], row["ly"])
                plt.text(center[0], center[1], row["room"], fontsize=10, ha='center', va='center', color='blue')
        
    def plot_points(self):
        if self.points_db is not None:
            for index, row in self.points_db.iterrows():
                plt.scatter(row["x"], row["y"], marker='o', facecolor='None', edgecolor='red', s=100)
        else:
            for index, row in self.rooms_db.iterrows():
                if not row["room"]=="extra_wall":
                    if int(row["room"][4])%2 == 0:
                        parity = "even" 
                    else:
                        parity = "odd"
                    for id in range(1,6):
                        pos = self._get_position(id, parity, self.door_pos_dict[row["room"]], row["refx"], row["refy"], row["lx"], row["ly"])
                        #print(pos, id, door_pos_dict[row["room"]])
                        plt.scatter(pos[0], pos[1], marker='o', facecolor='None', edgecolor='red', s=100) 

    def _save_points(self):
        points = []
        for index, row in self.rooms_db.iterrows():
            if not row["room"]=="extra_wall":
                if int(row["room"][4])%2 == 0:
                    parity = "even" 
                else:
                    parity = "odd"
                for id in range(1,6):
                    pos = self._get_position(id, parity, self.door_pos_dict[row["room"]], row["refx"], row["refy"], row["lx"], row["ly"])
                    points.append([row["room"], id, pos[0], pos[1]])
        points_db = pd.DataFrame(points, columns = ["room", "room_part", "x", "y"])
        points_db.to_csv("points_table.csv", index=False)

    def _save_rooms_with_polygon(self):
        polys = []
        for index, row in self.rooms_db.iterrows():
            vertices = self._get_polygon(row['refx'], row["refy"], row["lx"], row["ly"])
            polys.append(Polygon(vertices))
            
        rooms_gdb = gpd.GeoDataFrame(self.rooms_db, geometry=gpd.GeoSeries(polys))
        rooms_gdb.to_file("rooms_table.geojson", index=False)