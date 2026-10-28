"""Python 热身闯关 · 10 关（Day 2 上午 · 约 60-90 分钟）

这 10 关覆盖了 core.py 需要的**全部** Python 知识点，一关一个点。

玩法：
  1. 把每个函数里的 TODO 改成你的答案（基本一两行）；
  2. 运行本文件看通关进度：
       .venv/Scripts/python scripts/00_python_warmup.py
  3. 某关 ❌ 就改了再跑，直到 10/10 全绿。

规则：每一关先自己想 5 分钟；卡超过 10 分钟，带着你的报错/困惑去问 AI 讲解，
听完理解后回来**自己动手改出答案**——不要让 AI 直接给答案抄上。
"""

import numpy as np


# ---------- 第 1 关：字符串有多长 ----------
def e01():
    s = "企业知识库问答系统"
    n = len(s)  # TODO: 把 n 改成 s 的长度（用一个内置函数）
    assert n == 9


# ---------- 第 2 关：切片，取左段 ----------
def e02():
    s = "ABCDEFGHIJ"
    first3 = s[0:3]  # TODO: 取 s 的前 3 个字符（切片写法 s[a:b]，含头不含尾）
    assert first3 == "ABC"


# ---------- 第 3 关：带变量的切片 ----------
def e03():
    s = "ABCDEFGHIJ"
    start, size = 4, 3
    piece = s[start:size+start]  # TODO: 从下标 start 开始，取 size 个字符（下标 4 开始是 "EFG"）
    assert piece == "EFG"


# ---------- 第 4 关：切过头也不报错 ----------
def e04():
    s = "ABCDEFGHIJ"  # 只有 10 个字符
    piece = s[8:20]  # TODO: 从下标 8 开始想取 20 个字符（实际上只剩 2 个）
    # 提示：Python 切片越界不报错、能取多少给多少——
    # 这正是分块时"最后一块更短"的天然行为，core.py 会直接利用它
    assert piece == "IJ"


# ---------- 第 5 关：while 循环 + 手动推进 ----------
def e05():
    total = 0
    i = 1
    # TODO: 用 while 循环把 1+2+...+10 累加进 total
    # 提醒：循环体内必须推进 i，否则死循环（Ctrl+C 可以中断）
    while i<=10:
        total+=i
        i+=1
    assert total == 55


# ---------- 第 6 关：往列表里装东西 ----------
def e06():
    squares = []
    for x in [1, 2, 3, 4]:
        squares.append(x * x)  # TODO: 删掉 pass，把 x*x 追加进 squares（列表的方法 append）
    assert squares == [1, 4, 9, 16]


# ---------- 第 7 关：字典的取值 ----------
def e07():
    d = {"text": "一段分块内容", "start": 250}
    a = d["text"]  # 这就是按键取值
    b = d["start"] # TODO: 把 b 改成 d 里键为 "start" 的值
    assert (a, b) == ("一段分块内容", 250)


# ---------- 第 8 关：函数与 return ----------
def e08():
    def next_start(cur, chunk_size, overlap):
          # TODO: 返回"下一块的起点" = 当前起点 + chunk_size - overlap
          return cur + chunk_size - overlap
    assert next_start(0, 300, 50) == 250
    assert next_start(250, 300, 50) == 500


# ---------- 第 9 关：迷你分块（本关 = core.py 的完整骨架） ----------
def e09():
    text = "甲乙丙" * 4  # 12 个字符
    chunks = []
    start = 0
    # TODO：写一个 while 循环，循环体两件事：
    #   ① 把 text[start:start+5] 追加进 chunks        （第 3 关 + 第 6 关）
    #   ② start 推进 5 - 1 = 4                        （第 5 关 + 第 8 关）
    # 循环条件：start 小于字符串长度
    while start<len(text):
        chunks.append(text[start:start+5])
        start+=4
    assert chunks == ["甲乙丙甲乙", "乙丙甲乙丙", "丙甲乙丙"]


# ---------- 第 10 关：numpy 点积与模长 ----------
def e10():
    a = np.array([1.0, 0.0])
    b = np.array([3.0, 4.0])
    dot = a@b  # TODO: a 和 b 的点积（写法：a @ b）
    nb = np.linalg.norm(b)   # TODO: b 的模长（向量长度，写法：np.linalg.norm(b)）
    assert dot == 3.0
    assert abs(nb - 5.0) < 1e-9


# ================= 以下是自动判分，不用改 =================
EXERCISES = [e01, e02, e03, e04, e05, e06, e07, e08, e09, e10]

if __name__ == "__main__":
    passed = 0
    for idx, fn in enumerate(EXERCISES, 1):
        try:
            fn()
            print(f"✅ 第 {idx} 关 通过")
            passed += 1
        except AssertionError:
            print(f"❌ 第 {idx} 关 结果不对，改完再跑")
        except Exception as e:  # noqa: BLE001
            print(f"❌ 第 {idx} 关 报错：{type(e).__name__}: {e}")
    print(f"\n通关进度：{passed}/{len(EXERCISES)}")
    if passed == len(EXERCISES):
        print("全部通关！接下来打开 rag/core.py，把 4 个 TODO 填掉。")
