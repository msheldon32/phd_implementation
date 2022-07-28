import random
import util.cdf

class Driver:
    def __init__(self, n_regions, model):
        self.location = random.randrange(0, n_regions)
        self.status = "free"
        self.phi_vector = [0 for i in range(n_regions)]
        self.model = model
        self.destination = -1
    
    def transition_to_region(self, new_region):
        self.location = new_region
        if self.status == "free":
            if random.random() < self.phi_vector[new_region]:
                self.status = "busy"
                if self.model.get_region(new_region).remove_passenger():
                    t_prob = self.model.customer_destination_prob[new_region]
                    self.destination = util.cdf.generate_value(t_prob, random.random())
                else:
                    t_prob = self.model.customer_destination_prob[new_region]
                    self.destination = util.cdf.generate_value(t_prob, random.random())
            else:
                t_prob = self.model.free_destination_prob[new_region]
                self.destination = util.cdf.generate_value(t_prob, random.random())
        elif new_region == self.destination:
            self.destination = -1
            self.status = "free"
            self.transition_to_region(new_region)

    def transition_rate(self):
        return self.model.transition_rates[self.location][self.destination]
        

class Region:
    def __init__(self, model):
        self.passenger_queue = []
        self.model = model

    def passenger_arrival(self, p_type):
        self.passenger_queue.append(p_type)

    def has_passenger(self):
        return len(self.passenger_queue) == 0

    def remove_passenger(self):
        if self.has_passenger():
            del self.passenger_queue[0]
            return True
        return False

class Model:
    def __init__(self, n_regions, n_drivers):
        self.drivers = [Driver(n_regions, self) for i in n_drivers]
        self.regions = [Region(self) for i in n_regions]

        self.customer_destination_prob = [
            [(1/n_regions) for i in range(n_regions)]
                for j in range(n_regions)
            ]
        self.free_destination_prob = [
            [(1/n_regions) for i in range(n_regions)]
                for j in range(n_regions)
            ]


    def get_region(self, i):
        return self.regions[i]


if __name__=="__main__":
    pass
