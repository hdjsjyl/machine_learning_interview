import sys
res = 0
def helper(lists, m, idx, path):
    global res
    if idx >= len(lists):
        return

    if path%m > res:
        res = path%m
    # for i in range(idx, len(lists)):
    #     path += lists[i]
    #     helper(lists, m, i+1, path)
    #     path -= lists[i]
    path += lists[idx]
    helper(lists, m, idx+1, path)
    path -= lists[idx]
    helper(lists, m, idx+1, path)

if __name__ == "__main__":
    # input = sys.stdin.readline().strip()
    # input = '6 11'
    # if not input:
    #     print(0)
    # else:
    #     data = input.split(' ')
    #     n = int(data[0])
    #     m = int(data[1])
    #     # lists = sys.stdin.readline().strip()
    #     lists = '4 12 3 7 6 2'
    #     lists = lists.split(' ')
    #     lists = [ int(i)%m for i in lists]
    #     assert n == len(lists)
    #     # helper(lists, m, 0, 0)
    #     # print(res)
    #     dps = [0 for i in range(n)]
    #     dps[0] = lists[0]
    #     for i in range(1, n):
    #         dps[i] = max((dps[i-1]+lists[i])%m, dps[i-1])
    #         # print(i, dps)
    #     print(dps[-1])
    r = 'a b'
    a = r.split()
    print(a)