


if __name__ == "__main__":
    # Based on Wollenstein-Betech, this pricing model involves maximizing
    #     sum { lambda_{ij} (p_ij) - c_{ij} lambda_{ij} - c^c {lambda^0_{ij} - lambda_{ij}(u_ij}) }
    #     s.t. sum{lambda_{ij}} = sum{lambda_{ji}}

    # Our model does not feature operational costs per trip, or costs for lost customers due to unmet demand

    # thus we get

    #    max { sum {lambda_{ij} (p_ij)}}
    #     s.t. sum{lambda_{ij}} = sum{lambda_{ji}}


