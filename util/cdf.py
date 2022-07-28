def generate_value(pdf, prob):
    cm_prob = 0
    for i,p in enumerate(pdf):
        cm_prob += p
        if cm_prob >= prob:
            return i
    return -1
