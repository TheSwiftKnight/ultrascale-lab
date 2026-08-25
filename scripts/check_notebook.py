#!/usr/bin/env python3
"""
check_notebook.py — notebook 的靜態檢查：語法 + 名稱相依順序

Jupyter 是由上而下執行的，所以某一格用到的名稱必須在**該格或更早的格子**裡
定義過。這件事眼睛看不出來 —— §1.5 的續跑狀態用了 §4.1 才定義的 is_done()，
在 Colab 上跑到才炸 NameError，而且是在你已經等了幾分鐘掛完 Drive 之後。

用法：
    python3 scripts/check_notebook.py notebooks/week3_colab.ipynb

會回報：
  - 任何一格的 Python 語法錯誤（%magic 與 !shell 會先換成 pass）
  - 任何用到「還沒定義」名稱的格子

注意：函式主體裡的名稱是延遲解析的，理論上可以晚一點才定義。這裡仍然一併檢查，
因為「helper 定義在很後面」本身就是個壞味道，而且假警報可以直接看得出來。
"""
import ast, builtins, json, sys

BUILTIN = set(dir(builtins)) | {
    "__name__", "get_ipython", "display", "In", "Out", "exit", "quit",
}

class Collector(ast.NodeVisitor):
    def __init__(self):
        self.defined, self.used = set(), set()
    def visit_Import(self, n):
        for a in n.names: self.defined.add((a.asname or a.name).split('.')[0])
    def visit_ImportFrom(self, n):
        for a in n.names: self.defined.add(a.asname or a.name)
    def visit_FunctionDef(self, n):
        self.defined.add(n.name)
        for a in ast.walk(ast.arguments(**{f: getattr(n.args, f) for f in n.args._fields})):
            if isinstance(a, ast.arg): self.defined.add(a.arg)
        self.generic_visit(n)
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_ClassDef(self, n):
        self.defined.add(n.name); self.generic_visit(n)
    def visit_Lambda(self, n):
        for a in n.args.args + n.args.kwonlyargs: self.defined.add(a.arg)
        if n.args.vararg: self.defined.add(n.args.vararg.arg)
        if n.args.kwarg:  self.defined.add(n.args.kwarg.arg)
        self.generic_visit(n)
    def visit_ExceptHandler(self, n):
        if n.name: self.defined.add(n.name)
        self.generic_visit(n)
    def visit_Global(self, n):
        self.defined.update(n.names)
    def visit_Name(self, n):
        (self.defined if isinstance(n.ctx, (ast.Store, ast.Del)) else self.used).add(n.id)
    def visit_alias(self, n):
        self.defined.add(n.asname or n.name)
    def visit_comprehension(self, n):
        self.generic_visit(n)

def strip_magic(src):
    out = []
    for l in src.split('\n'):
        st = l.strip()
        out.append(' ' * (len(l) - len(l.lstrip())) + 'pass' if (st.startswith('%') or st.startswith('!')) else l)
    return '\n'.join(out)

nb = json.load(open(sys.argv[1]))
known, problems = set(BUILTIN), []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code': continue
    src = strip_magic(''.join(c['source']))
    try: tree = ast.parse(src)
    except SyntaxError as e:
        problems.append((i, f"語法錯誤 L{e.lineno}: {e.msg}")); continue
    col = Collector(); col.visit(tree)
    title = ''.join(c['source']).split('\n')[0].replace('#@title', '').strip()
    missing = sorted(n for n in col.used - col.defined - known)
    if missing:
        problems.append((i, f"用到還沒定義的名稱 {missing}  ← {title}"))
    known |= col.defined

# ---- 檢查 2：標題編號必須和實際位置一致 ----
# §1.5 曾經被放在 §3.3 和 §4 中間，標題寫 1.5 但實際在第 21 格 ——
# 使用者照標題找當然找不到。編號應該隨位置遞增。
import re
order, last, seq_problems = [], None, []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code': continue
    first = ''.join(c['source']).split('\n')[0]
    m = re.match(r'\s*#@title\s+(\d+)\.(\d+)([a-z]?)', first)
    if not m: continue
    key = (int(m.group(1)), int(m.group(2)), m.group(3) or '')
    label = f"{m.group(1)}.{m.group(2)}{m.group(3)}"
    if last and key < last[0]:
        seq_problems.append(f"  ❌ cell {i}: 標題 §{label} 排在 §{last[1]}（cell {last[2]}）後面，編號倒退了")
    if not last or key > last[0]:
        last = (key, label, i)
    order.append((i, label))

for i, msg in problems: print(f"  ❌ cell {i}: {msg}")
for m in seq_problems: print(m)

ok = not problems and not seq_problems
print(f"\n{'✅ 語法、相依順序、標題編號全部正確' if ok else f'{len(problems)+len(seq_problems)} 個問題'}")
if ok:
    print(f"   （{len(order)} 個編號格子：{order[0][1]} → {order[-1][1]}）")
sys.exit(0 if ok else 1)
