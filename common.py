# ----------------------- Thor -------------------------------

THOR_SAMPLING_RATE = 3
REGION = [(95, 40), (87, 18), (42,3), (3, 20),(55, 40), (15, 36)]
REGION_ACCEPT_DIST = [7, 7, 10, 12, 3, 5]

REGION_ACCEPT_DIST_LAB = [10, 7, 12, 12, 10, 5]
REGION_ACCEPT_DIST_2 = [7, 8, 10, 12, 3, 5]
REGION_ACCEPT_DIST_4 = [7, 8, 9, 12, 2, 4]
REGION_ACCEPT_DIST_5 = [7, 8, 7.1, 12, 2, 4]
REGION_ACCEPT_DIST_6 = [7, 8, 8, 12, 2, 4]
REGION_ACCEPT_DIST_7 = [7, 8, 8, 12, 2, 4]
REGION_ACCEPT_DIST_8 = [7, 8, 10, 12, 2, 4]
REGION_ACCEPT_DIST_UTILITY = [7, 8, 10, 12, 8, 4]
REGION_ACCEPT_DIST_INSPECTOR = [7, 8, 10, 12, 10, 7]

# ------------------------------------------------------------


def fill_in_the_gap(observation):
    [observation.pop(idx) for idx, _ in enumerate(observation) if idx < len(observation)-1 and observation[idx] == observation[idx+1]]

    cnt = 0
    for idx, p in enumerate(observation[1:]):
        current_x, current_y = p[0], p[1]
        s_x, s_y = 1, 1
        prev_x, prev_y = observation[cnt+idx][0], observation[cnt+idx][1]
        if current_x < prev_x:
            s_x = -1
        if current_y < prev_y:
            s_y = -1        

        if prev_x == current_x:
            xlist = [prev_x]
        else:
            xlist = list(range(prev_x, current_x, 1*s_x))
        if prev_y == current_y:
            ylist = [prev_y]          
        else:
            ylist = list(range(prev_y, current_y, 1*s_y))

        xlist_rightmost = xlist[-1]
        ylist_rightmost = ylist[-1]    

        while len(xlist) != len(ylist):
            if len(xlist) < len(ylist):
                xlist.insert(len(xlist), xlist_rightmost)
            if len(ylist) < len(xlist):
                ylist.insert(len(ylist), ylist_rightmost)                
        observation.pop(cnt+idx)
        [observation.insert(cnt+idx+i, x) for i, x in enumerate(list(zip(xlist, ylist)))]
        cnt += len(list(zip(xlist, ylist)))-1
    return observation