import csv

curClass = ""
curIndex = -2
bs1 = []
bs2 = []

classes = [
    "smart_1",
    "cheops",
    "lisa_pathfinder",
    "debris",
    "proba_3_ocs",
    "proba_3_csc",
    "soho",
    "earth_observation_sat_1",
    "proba_2",
    "xmm_newton",
    "double_star",
]

with open('bababooey.csv', newline='') as trainout:
    
    params = csv.reader(trainout, delimiter=',')

    i = 0
    for row in params:
        if curClass != row[1] and row[1] != "class":
            curClass = row[1]
            curIndex = 0
#        if row[1] != "class":
#            row[1]=classes.index(row[1])
#        bs.append(row)
        if curIndex < 2000:
            bs1.append(row)
        else:
            bs2.append(row)
        curIndex += 1
   
    print("done!")

with open('newval.csv', "a", newline='') as trainout:
    writer = csv.writer(trainout, delimiter=",")
    writer.writerows(bs1)

with open('newtrain.csv', "a", newline='') as trainout:
    writer = csv.writer(trainout, delimiter=",")
    writer.writerows(bs2)