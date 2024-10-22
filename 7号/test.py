def max_pattern_occurrences(s, t):
    # 统计 s 和 t 中每个字符的频率
    count_s = [0] * 26  # 统计 s 中字符的频率
    count_t = [0] * 26  # 统计 t 中字符的频率

    # 统计 s 中每个字符的频率
    for char in s:
        count_s[ord(char) - ord('a')] += 1

    # 统计 t 中每个字符的频率
    for char in t:
        count_t[ord(char) - ord('a')] += 1

    # 检查是否存在 t 中某个字符在 s 中的数量不足的情况
    for i in range(26):
        if count_t[i] > 0 and count_s[i] < count_t[i]:
            return 0  # 如果 s 中某个字符不够用来匹配 t 的所有字符，返回 0

    # 定义最优的周期重复次数 k
    # 找到在 t 的周期内，每个字符出现次数的最小值，以便找到最大的重复周期数
    max_repeats = float('inf')
    for i in range(26):
        if count_t[i] > 0:  # 只考虑 t 中存在的字符
            # 计算字符 i 的最多周期重复次数
            max_repeats = min(max_repeats, count_s[i] // count_t[i])

    # 返回最优重复周期数
    return max_repeats


# 测试输入输出
def main():
    z = int(input())  # 读取测试集数量
    results = []
    for _ in range(z):
        s, t = input().split()
        results.append(str(max_pattern_occurrences(s, t)))

    # 输出所有测试集的结果
    print("\n".join(results))

 b
if __name__ == "__main__":
    main()
