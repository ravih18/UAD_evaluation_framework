def get_closest_img_part(sub_ses, participants, dist, n=5, interval=1):
    index = participants.index(sub_ses)
    indexes = sorted(range(len(dist[index])), key=lambda k: dist[index][k])
    i, closest_part, scores = 0, [], []
    while len(closest_part) < n:
        if participants[indexes[i]][0] != sub_ses[0]:
            closest_part.append(participants[indexes[i]])
            scores.append(dist[index][indexes[i]])
        i+=interval
    return closest_part, scores

def get_closest_img_same_part(sub_ses, participants, dist):
    """
    """
    index = participants.index(sub_ses)
    indexes = sorted(range(len(dist[index])), key=lambda k: dist[index][k])
    closest_part, scores = [], []
    #print(indexes)
    for i in range(len(indexes)):
        if participants[indexes[i]][0] == sub_ses[0]:
            closest_part.append(participants[indexes[i]])
            scores.append(dist[index][indexes[i]])
    return scores