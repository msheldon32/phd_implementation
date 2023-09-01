import random

class Model:
    def __init__(self, controller):
        self.cells = {}
        self.vehicles = []
        self.controller = controller

        self.request_queue = []

        self.profit = 0

    def set_shortest_paths(self):
        for cell_id, cell in self.cells.items():
            # find all shortest paths from [cell_id] using Dijkstra's algorithm
            dist = [float('inf')] * len(self.cells)
            prev = [None] * len(self.cells)
            dist[cell_id] = 0
            queue = [cell_id]

            while queue:
                u = queue.pop(0)
                for next_cell, p in self.cells[u].get_adjacent_cells():
                    alt = dist[u] + (1/p)
                    if alt < dist[v]:   
                        dist[v] = alt
                        prev[v] = u 
                        queue.append(v)

            # to do: finish up this algorithm and do set_next_cell accordingly

    def add_cell(self, cell):
        self.cells.append(cell)

    def add_vehicle(self, vehicle):
        self.vehicles.append(vehicle)
        vehicle.update_location(random.choice(self.cells.keys()))

    def get_next_cell(self, src, dst):
        return self.cells[src].get_next_cell(dst)

    def step(self):
        for vehicle in self.vehicles:
            if vehicle.location == vehicle.get_destination():
                vehicle.finish_trip()
            else:
                vehicle.location = self.get_next_cell(vehicle.location, vehicle.get_destination())
