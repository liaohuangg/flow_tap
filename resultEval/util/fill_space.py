import os, math, sys
from operator import itemgetter, attrgetter
# with open('path.txt','r') as PATHFILE:
# 	path = PATHFILE.readline().split()[0]
# path = 'results/test_conti_fix_interposer/'
# path = ''


class FlpItem:
    def __init__(self, name, width, height, x, y):
        self.name = name
        self.width = width
        self.height = height
        self.x = x
        self.y = y

    def __repr__(self):
        return repr((self.name, self.width, self.height, self.x, self.y))


def fill_space(width_st, width_ed, height_st, height_ed, filesim, filein, fileout):
    UnderFill = "\t2.32E+06\t0.625\n"
    ws = []
    ws_n = 0
    sep_n = 16

    def cut_vertical(cur_list, xst, xed, yst, yed):
        nonlocal ws_n, sep_n, ws
        if xed - xst < 0.00001 or yed - yst < 0.00001:
            return
        if cur_list == []:
            ws.append(FlpItem('WS_' + str(ws_n), xed - xst, yed - yst, xst, yst))
            ws_n += 1
            return
        if len(cur_list) == 1:
            i = cur_list[0]
            if i.x - xst < 0.00001 and i.y - yst < 0.00001 and xed - (i.x + i.width) < 0.00001 and yed - (i.y + i.height) < 0.00001:
                return
        cur_list = sorted(cur_list, key=attrgetter('y', 'x'))
        cutlines = [xst]
        for i in cur_list:
            if i.y - yst < 0.00001:
                if i.x - xst >= 0.00001:
                    cutlines.append(i.x)
                if xed - (i.x + i.width) >= 0.00001:
                    cutlines.append(i.x + i.width)
        cutlines.append(xed)
        cutlines = sorted(list(set(cutlines)))

        if len(cutlines) == 2:
            if cur_list[0].y - yst < 0.00001:
                next_y = cur_list[0].y + cur_list[0].height
                if next_y - yst < 0.00001 or yed - next_y < 0.00001:
                    return
                remaining = [i for i in cur_list if (i.y + i.height) - next_y > -0.00001 and yed - i.y > -0.00001]
                if not remaining:
                    return
                cut_vertical(remaining, xst, xed, next_y, yed)
                return
            ws.append(FlpItem('WS_' + str(ws_n), xed - xst, cur_list[0].y - yst, xst, yst))
            ws_n += 1
            cut_vertical(cur_list, xst, xed, cur_list[0].y, yed)
        else:
            for l in range(1, len(cutlines)):
                left_list = []
                right_list = []
                for i in cur_list:
                    if (i.x + i.width) - cutlines[l] < 0.00001:
                        left_list.append(i)
                    elif (cutlines[l] - i.x >= 0.00001) and ((i.x + i.width) - cutlines[l] >= 0.00001):
                        left_list.append(FlpItem('Unit_' + str(sep_n), cutlines[l] - i.x, i.height, i.x, i.y))
                        right_list.append(FlpItem('Unit_' + str(sep_n + 1), i.x + i.width - cutlines[l], i.height, cutlines[l], i.y))
                        sep_n += 2
                    else:
                        right_list.append(i)

                cur_list = right_list
                cut_horizontal(left_list, cutlines[l - 1], cutlines[l], yst, yed)

    def cut_horizontal(cur_list, xst, xed, yst, yed):
        nonlocal ws_n, sep_n, ws
        if xed - xst < 0.00001 or yed - yst < 0.00001:
            return
        if cur_list == []:
            ws.append(FlpItem('WS_' + str(ws_n), xed - xst, yed - yst, xst, yst))
            ws_n += 1
            return
        if len(cur_list) == 1:
            i = cur_list[0]
            if i.x - xst < 0.00001 and i.y - yst < 0.00001 and xed - (i.x + i.width) < 0.00001 and yed - (i.y + i.height) < 0.00001:
                return
        cur_list = sorted(cur_list, key=attrgetter('x', 'y'))
        cutlines = [yst]
        for i in cur_list:
            if i.x - xst < 0.00001:
                if i.y - yst >= 0.00001:
                    cutlines.append(i.y)
                if yed - (i.y + i.height) >= 0.00001:
                    cutlines.append(i.y + i.height)
        cutlines.append(yed)
        cutlines = sorted(cutlines)

        if len(cutlines) == 2:
            if cur_list[0].x - xst < 0.00001:
                next_x = cur_list[0].x + cur_list[0].width
                if next_x - xst < 0.00001 or xed - next_x < 0.00001:
                    return
                remaining = [i for i in cur_list if (i.x + i.width) - next_x > -0.00001 and xed - i.x > -0.00001]
                if not remaining:
                    return
                cut_horizontal(remaining, next_x, xed, yst, yed)
                return
            ws.append(FlpItem('WS_' + str(ws_n), cur_list[0].x - xst, yed - yst, xst, yst))
            ws_n += 1
            cut_horizontal(cur_list, cur_list[0].x, xed, yst, yed)
        else:
            for l in range(1, len(cutlines)):
                down_list = []
                up_list = []
                for i in cur_list:
                    if (i.y + i.height) - cutlines[l] < 0.00001:
                        down_list.append(i)
                    elif (cutlines[l] - i.y >= 0.00001) and ((i.y + i.height) - cutlines[l] >= 0.00001):
                        down_list.append(FlpItem('Unit_' + str(sep_n), i.width, cutlines[l] - i.y, i.x, i.y))
                        up_list.append(FlpItem('Unit_' + str(sep_n + 1), i.width, i.y + i.height - cutlines[l], i.x, cutlines[l]))
                        sep_n += 2
                    else:
                        up_list.append(i)

                cur_list = up_list
                cut_vertical(down_list, xst, xed, cutlines[l - 1], cutlines[l])

    def print_list(p):
        for i in p:
            print(i)
        print()

    flplist = []
    with open(filesim + '.flp', 'r') as FlpIn:
        n = 0
        for line in FlpIn:
            sp = line.split()
            if sp:
                if sp[0] != '#':
                    flplist.append(FlpItem(sp[0], float(sp[1]), float(sp[2]), float(sp[3]), float(sp[4])))
                    n += 1

    cut_vertical(flplist, width_st, width_ed, height_st, height_ed)

    with open(fileout + '.flp', 'w') as FlpOut:
        with open(filein + '.flp', 'r') as FlpIn:
            for line in FlpIn:
                FlpOut.write(line)
        for i in range(0, ws_n):
            FlpOut.write(ws[i].name + '\t' + str(ws[i].width) + '\t' + str(ws[i].height) + '\t' + str(ws[i].x) + '\t' + str(ws[i].y) + UnderFill)


if __name__ == "__main__":
    fill_space()
