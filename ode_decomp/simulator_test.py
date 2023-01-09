import simulator 

def test_simulator():
    stn_to_cell = [0,0,1,1]
    cell_to_stn = [set([0,1]), set([2,3])]
    n_cells = 2
    stations = [0,1,2,3]

    mu = [[
        [  # starting from cell 1
          [ # going to cell 1
           [2, 5], [10, 20]
            ],
          [ # going to cell 2
            [8,8,8,8], [14,7]
            ]
         ],
        [  # starting from cell 2
          [ # going to cell 1
           [2, 10], [10, 10]
            ],
          [ # going to cell 2
            [7, 15], [7,7,7,7,7],
            ]
         ], 
        ]]

    phi = [[
        [  # starting from cell 1
          [ # going to cell 1
           [0.5, 1.0], [0.1, 1.0]
            ],
          [ # going to cell 2
            [0,0,0,1.0], [0.5,1.0]
            ]
         ],
        [  # starting from cell 2
          [ # going to cell 1
           [0.2, 1.0], [10, 1.0]
            ],
          [ # going to cell 2
            [0.3, 1.0], [0,0,0,0,1.0],
            ]
         ], 
        ]]

    in_demands = []

    in_probabilities = [[
        [  # going to cell 1
         [ # starting from cell 1
          [0.3, 0.7], # starting from station 1
          [0.5, 0.5]  # starting from station 2
         ],
         [ # starting from cell 2
          [0.1, 0.9], # starting from station 3
          [0.3, 0.7]  # starting from station 4
         ],
        ],
        [  # going to cell 2
         [ # starting from cell 1
          [0.8, 0.2], # starting from station 1
          [0.6, 0.4]  # starting from station 2
         ],
         [ # starting from cell 2
          [0.4, 0.6], # starting from station 3
          [0.2, 0.8]  # starting from station 4
         ],
        ],
        ]]

    out_demands = [[
        [  # starting from cell 1
         [ # starting from station 1
          [3, 6], # going to cell 1
          [5, 7] # going to cell 2
         ],
         [ # starting from station 2 
          [1, 5], # going to cell 1
          [3, 4] # going to cell 2
         ],
        ],
        [  # starting from cell 2 
         [ # starting from station 3
          [1, 2], # going to cell 1
          [3, 4] # going to cell 2
         ],
         [ # starting from station 4 
          [5, 6], # going to cell 1
          [7, 8] # going to cell 2
         ],
        ],
        ]]

    starting_levels = [5,5]
    prices = [1.0,1.0]

    
    sim = simulator.Simulator (stn_to_cell, cell_to_stn, n_cells, stations, mu, phi, in_demands, in_probabilities, out_demands, starting_levels, prices)

    t = 0

    while not sim.is_finished(t):
        hour_idx = sim.get_hour_idx(t)
        stn_rates, cum_stn_rate, delay_rates, cum_delay_rate = sim.get_rates(hour_idx)

        t = sim.get_next_time(t, cum_stn_rate + cum_delay_rate)

