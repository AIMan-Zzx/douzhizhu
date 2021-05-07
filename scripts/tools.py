from utils import to_char, to_value, get_mask_alter, give_cards_without_minor, \
    get_mask, action_space_single, action_space_pair, get_category_idx, normalize

cards = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', '*', '$']

class SQITCards():
    def __init__(self,cnts=5):
        self.ctns = cnts

    def remove_card(self,cards_set,low_card):
        # return cards_set
        new_card_sets = []
        low_index = cards.index(low_card)
        for card in cards_set:
            card_index = cards.index(card)
            if card_index > low_index:
                new_card_sets.append(card)
        return new_card_sets

    def guess_cards(self,cards_set):
        cards_set = self.remove_card(cards_set,'6')
        sets = []
        if self.ctns == 1:
            cards_set = self.remove_card(cards_set, '10')
            l = len(cards_set)
            for i in range(l):
                one = cards_set[i]
                set = [one]
                set = sorted(set)
                # print(set)
                if set not in sets:
                    # print(set)
                    sets.append(set)

        if self.ctns == 2:
            cards_set = self.remove_card(cards_set, '9')
            l = len(cards_set)
            for i in range(l):
                for j in range(i + 1, l):
                    one = cards_set[i]
                    two = cards_set[j]
                    set = [one, two]
                    set = sorted(set)
                    # print(set)
                    if set not in sets:
                        # print(set)
                        sets.append(set)

        if self.ctns == 3:
            cards_set = self.remove_card(cards_set, '8')
            l = len(cards_set)
            for i in range(l):
                for j in range(i + 1, l):
                    for k in range(j + 1, l):
                        one = cards_set[i]
                        two = cards_set[j]
                        three = cards_set[k]
                        set = [one, two, three]
                        set = sorted(set)
                        # print(set)
                        if set not in sets:
                            # print(set)
                            sets.append(set)

        if self.ctns == 4:
            cards_set = self.remove_card(cards_set, '7')
            l = len(cards_set)
            for i in range(l):
                for j in range(i + 1, l):
                    for k in range(j + 1, l):
                        for m in range(k + 1, l):
                            one = cards_set[i]
                            two = cards_set[j]
                            three = cards_set[k]
                            four = cards_set[m]
                            set = [one, two, three, four]
                            set = sorted(set)
                            # print(set)
                            if set not in sets:
                                # print(set)
                                sets.append(set)
        elif self.ctns == 5:
            l = len(cards_set)
            for i in range(l):
                for j in range(i + 1, l):
                    for k in range(j + 1, l):
                        for m in range(k + 1, l):
                            for n in range(m + 1, l):
                                one = cards_set[i]
                                two = cards_set[j]
                                three = cards_set[k]
                                four = cards_set[m]
                                five = cards_set[n]

                                set = [one, two, three, four,five]
                                set = sorted(set)
                                # print(set)
                                if set not in sets:
                                    # print(set)
                                    sets.append(set)
        return sets

def make_sqit_cards(cnts):
    return SQITCards(cnts)
