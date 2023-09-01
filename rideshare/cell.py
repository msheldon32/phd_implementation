
class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vehicles = set()
        self.origin_rate = 0
        self.destination_rate = 0

        self.adjacent_cells = {}
        self.next_cells = {}

    def add_vehicle(self, vehicle):
        self.vehicles.append(vehicle)

    def remove_vehicle(self, vehicle):
        self.vehicles.remove(vehicle)

    def set_origin_rate(self, rate):
        self.origin_rate = rate

    def set_destination_rate(self, rate):
        self.destination_rate = rate

    def add_adjacent_cell(self, cell, rate):
        self.adjacent_cells[cell] = rate

    def get_adjacent_cells(self):
        return self.adjacent_cells

    def set_next_cell(self, dst_cell, next_cell):
        """
            Given a destination cell, set the next cell to go to for the shortest path
        """
        self.next_cells[dst_cell] = next_cell

    def get_next_cell(self, dst_cell):
        """
            Given a destination cell, return the next cell to go to for the shortest path
        """
        return self.next_cells[dst_cell]
