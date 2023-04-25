from gt_analysis import TputData, CostGenerator

import pytest

TOL = 0.002

def test_tput_concavity():
    tput_data = TputData()

    # Check the following:
    #  1. Throughput is 0 at 0
    #  2. Throughput is monotonically increasing
    #  3. Throughput is concave
    #  4. Throughput is anti-starshaped

    # all of these should be proved by Dowdy et al., and Shanthikumar&Yao
    for run_no in range(tput_data.n_runs):
        tput = tput_data.run_data[run_no]

        assert -TOL <= tput[0] <= TOL

        first_differences = [tput[i+1] - tput[i] for i in range(len(tput)-1)]
        second_differences = [first_differences[i+1] - first_differences[i] for i in range(len(first_differences)-1)]

        assert all([diff >= -TOL for diff in first_differences])
        assert all([diff <= TOL for diff in second_differences])

        #starshape_differences = [(tput[i+1]/(i+2)) - (tput[i]/(i+1)) for i in range(len(tput-1))]
        starshape_differences = [(tput[i+1]/(i+1)) - (tput[i]/(i)) for i in range(len(tput)-1) if i > 0]
        assert all([diff <= TOL for diff in starshape_differences])

def test_rel_concavity():
    tput_data = TputData()

    # 1. Check concavity and monotonicity of throughput for user x
    # 2. Check intermediate proof steps
    #      a. The second difference of the relative throughput is equal to the second difference of the throughput minus x_{-r} * Z(x_{-r} + x)
    #      b. The second difference of the relative throughput is equal to twice the first difference of Z plus (x+1) times the second difference of Z
    for run_no in range(tput_data.n_runs):
        tput = tput_data.run_data[run_no]

        for x_mr in range(0, tput_data.n_jobs):
            tput_conversion = lambda i: tput[i+x_mr] * (i/(i+x_mr))
            first_differences = [tput_conversion(i+1) - tput_conversion(i) for i in range(len(tput)-1) if i > x_mr and (i+x_mr) < len(tput)-1]
            second_differences = [first_differences[i+1] - first_differences[i] for i in range(len(first_differences)-1)]

            assert all([diff >= -TOL for diff in first_differences])
            assert all([diff <= TOL for diff in second_differences])

def test_dsc():
    tput_data = TputData()

    # 1. Check that the relative throughput is decreasing in x_{-r}
    # 2. Check that the first difference of relative throughput is decreasing in x_{-r}

    for run_no in range(tput_data.n_runs):
        tput = tput_data.run_data[run_no]

        first_difference_list = []

        for x_mr in range(0, tput_data.n_jobs):
            tput_conversion = lambda i: tput[i+x_mr] * (i/(i+x_mr))

            first_differences = [tput_conversion(i+1) - tput_conversion(i) for i in range(len(tput)-1) if i > x_mr and (i+x_mr) < len(tput)-1]

            first_difference_list.append(first_differences)

        for x_r in range(0, tput_data.n_jobs):
            prev = -1
            for x_mr in range(1, x_r):
                tput_conversion = lambda i: tput[i+x_mr] * (i/(i+x_mr))
                if x_r >= len(first_difference_list[x_mr]):
                    break
                assert first_difference_list[x_mr][x_r] <= first_difference_list[x_mr][x_r-1] + TOL
                
                if prev != -1:
                    assert tput_conversion(x_r) <= prev + TOL
                    prev = tput_conversion(x_r)
