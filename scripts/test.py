cards_set = [3,3,4,4,5,5,6,7,8,9,10,10,11,11,12,12,13,]
l = len(cards_set)
sets = []

for i in range(l):
    for j in range(i+1,l):
        for k in range(j+1,l):
            for m in range(k + 1, l):
                one = cards_set[i]
                two = cards_set[j]
                three = cards_set[k]
                four = cards_set[m]
                set = [one,two,three,four]
                set = sorted(set)
                # print(set)
                if set not in sets:
                    print(set)
                    sets.append(set)
print(len(sets))
