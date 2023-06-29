import csv
import random

N_JOBS = 40
FILE = "single_class_combined_2.csv"

class TputData:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(TputData, cls).__new__(cls)
        
            cls._instance.run_data = {}

            with open(FILE) as csvfile:
                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    run_no = int(row[0]) - 1
                    if run_no not in cls._instance.run_data:
                        cls._instance.run_data[run_no] = [-1] * (N_JOBS+1)
                        cls._instance.run_data[run_no][0] = 0
                    cls._instance.run_data[run_no][int(row[1])] = float(row[2])
            

            cls._instance.n_runs = len(cls._instance.run_data)
            cls._instance.n_jobs = N_JOBS

        return cls._instance
    
    def get_tput_at_job(cls, run_no, n_jobs):
        return cls._instance.run_data[run_no][n_jobs]

class CostGenerator:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(CostGenerator, cls).__new__(cls)
        return cls._instance

    def generate_cost_curve(cls, curve_type, length):
        if curve_type == "linear":
            slope = random.uniform(0.1, 0.5)
            return [slope * i for i in range(length)]
        elif curve_type == "exp":
            rate = random.uniform(1.1, 1.5)
            first_curve = [rate ** i for i in range(length)]
            return [x - first_curve[0] for x in first_curve]
        elif curve_type == "convex":
            # generate a random convex curve
            curve = [0]
            forward_diff = (random.uniform(1, 5)**2)/100

            for i in range(1, length):
                curve.append(curve[i-1] + forward_diff)
                forward_diff += (random.uniform(1, 5)**2)/100

            return curve
        elif curve_type == "zero":
            return [0] * length
        else:
            raise ValueError("Invalid curve type")
