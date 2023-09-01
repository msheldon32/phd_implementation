
class Vehicle:
    def __init__(self):
        self.location_queue = []
        self.location = None

    def update_location(self, location):
        self.location = location
        self.location_queue.append(location)

    def get_destination(self):
        return self.location_queue[0]

    def finish_trip(self):
        self.location_queue.pop(0)
