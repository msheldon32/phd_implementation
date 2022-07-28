import random
import util.cdf
import math

class Passenger:
    def __init__(self):
        self.reservation_value = random.randrange(2, 50)

    def offer_price(self, price):
        return price <= self.reservation_value

class Driver:
    def __init__(self):
        self.reservation_wage = random.randrange(5, 30)

    def offer_price(self, price, expected_time):
        return price >= self.reservation_wage

class Model:
    def __init__(self):
        self.waiting_queue = []
        self.busy_queue = []
        self.total_revenue = 0
        self.platform_ptg = 0.1
        self.driver_ptg = 0.9
        self.ex_passenger_arrival_rate = 10
        self.busy_rate = 10  # service rate of busy period queue
        self.ex_driver_arrival_rate = 1  # expected time

        self.t = 0

        self.exit_prob = 0.1

        expected_n_trips = 1/self.exit_prob
        self.ex_time =  expected_n_trips/self.busy_rate # fix this - expected number of trips * expected time per trip

    def get_price(self):
        return random.randrange(0, 50)

    def add_passenger(self, passenger):
        if len(self.waiting_queue) > 0:
            price = self.get_price()
            if passenger.offer_price(price):
                driver = self.waiting_queue[0].pop(0)
                self.busy_queue.append(driver)
                driver.add_pay(self.driver_ptg * price)
                self.total_revenue += self.platform_ptg * price

    def step(self):
        event_rates = [self.ex_driver_arrival_rate,
                       self.ex_passenger_arrival_rate] +
                      [self.busy_rate]
        total_event_rate = sum(event_rates)
        self.t += -(1/total_event_rate)*math.log(random.random())

        event_probs = [r/total_event_rate for r in event_rates]

        next_event = util.cdf.generate_value(event_probs, random.random())

        if next_event == 0:
            # driver arrival
            new_driver = Driver()
            if driver.offer_price(self.get_price(), self.ex_time):
                self.waiting_queue.append(new_driver)
        elif next_event == 1:
            # customer arrival
            new_customer = Customer()
            if customer.offer_price(self.get_price()):
                if len(self.waiting_queue) == 0:
                    last_driver = self.waiting_queue.pop(0)
                    self.busy_queue.append(last_driver)
        else:
            # remove a driver from the busy queue
            last_driver = self.busy_queue.pop(0)
            if random.random() < self.exit_prob:
                pass
            else:
                self.waiting_queue.append(last_driver)
