"""Rapfi 玩家（Piskvork/pbrain 协议）—— 给 train.py 的评估当对手用

接口对齐 game.py 的 Game.start_play：
    player.set_player_ind(p)    # p ∈ {1,2}
    player.get_action(board)    # 返回 move = r*15+c

三个坑（都是老项目实测踩过的）：
  1) BOARD 行是 `x,y,who`（先列后行！），不是 (r,c)
  2) who 的含义有争议：老项目实测「相对引擎」(自己=1/对手=2)；官方协议文档写「绝对颜色」。
     用 who_mode 参数切换，冒烟测试决定用哪个。
  3) 进程要常驻：每手重启会被 EOF 逼出仓促着法；每局之间用 END 收尾即可。
"""
import os
import select
import subprocess
import time


class RapfiError(RuntimeError):
    pass


class RapfiPlayer:
    def __init__(self, binary='~/rapfi_run/pbrain-rapfi', depth=1, size=15,
                 rule=0, timeout_turn=2000, who_mode='relative', use_rosetta=True,
                 nice=19, extra_args=None):
        self.binary = os.path.expanduser(binary)
        self.dir = os.path.dirname(self.binary)
        self.size = size
        self.rule = rule
        self.depth = depth
        self.who_mode = who_mode
        self.timeout_turn = timeout_turn
        self.use_rosetta = use_rosetta
        self.nice = nice
        self.extra_args = extra_args or []
        self.player = 1
        self.p = None
        self._start()

    # ---------- 进程与协议 ----------
    def _config_for_depth(self):
        """复制 config.toml → config_d{N}.toml，只改 [search] 段的 max_search_depth"""
        src = os.path.join(self.dir, 'config.toml')
        dst = os.path.join(self.dir, f'config_d{self.depth}.toml')
        if os.path.exists(dst):
            return os.path.basename(dst)
        with open(src) as f:
            lines = f.readlines()
        out, section = [], None
        for ln in lines:
            s = ln.strip()
            if s.startswith('[') and s.endswith(']'):
                section = s[1:-1].strip()
            if section == 'search' and s.replace(' ', '').startswith('max_search_depth'):
                indent = ln[:len(ln) - len(ln.lstrip())]
                out.append(f'{indent}max_search_depth = {self.depth}\n')
            else:
                out.append(ln)
        with open(dst, 'w') as f:
            f.writelines(out)
        return os.path.basename(dst)

    def _start(self):
        cfg = self._config_for_depth()
        cmd = []
        if self.nice:
            cmd += ['nice', '-n', str(self.nice)]
        if self.use_rosetta:
            cmd += ['arch', '-x86_64']
        cmd += ['./' + os.path.basename(self.binary), f'--config={cfg}'] + self.extra_args
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  cwd=self.dir)
        self._drain_banner(timeout=6.0)
        self._send(f'START {self.size}')
        self._read(timeout=10)
        self._send(f'INFO rule {self.rule}')
        self._send(f'INFO timeout_turn {self.timeout_turn}')
        self._send('INFO timeout_match 0')

    def _drain_banner(self, timeout=6.0):
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([self.p.stdout], [], [], 0.25)
            if not r:
                break
            if not self.p.stdout.readline():
                break

    def _send(self, text):
        if self.p is None or self.p.poll() is not None:
            raise RapfiError('engine 进程不在')
        try:
            self.p.stdin.write(text + '\n')
            self.p.stdin.flush()
        except Exception as e:
            raise RapfiError(f'写入失败: {e}')

    def _read(self, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([self.p.stdout], [], [], 0.5)
            if r:
                line = self.p.stdout.readline()
                if not line:
                    return None
                line = line.strip()
                if line and not line.startswith('MESSAGE'):
                    return line
            if self.p.poll() is not None:
                return None
        return None

    # ---------- 对外接口 ----------
    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        stones = [(int(m) // self.size, int(m) % self.size, int(q))
                  for m, q in board.states.items()]
        my_color = self.player
        lines = ['BOARD']
        for r, c, col in stones:
            if self.who_mode == 'relative':
                who = 1 if col == my_color else 2
            else:
                who = col                              # 绝对：1=黑 2=白
            lines.append(f'{c},{r},{who}')            # ← 先列后行
        lines.append('DONE')
        self._send('\n'.join(lines))
        resp = self._read(timeout=120)
        if not resp:
            raise RapfiError('无回应')
        parts = resp.split(',')
        if len(parts) < 2:
            raise RapfiError(f'异常回应: {resp!r}')
        try:
            x, y = int(parts[0]), int(parts[1])
        except ValueError:
            raise RapfiError(f'解析失败: {resp!r}')
        if x < 0 or y < 0:
            raise RapfiError(f'引擎认输/无着: {resp!r}')
        mv = y * self.size + x                   # (col,row) → move
        if mv not in board.availables:
            raise RapfiError(f'非法/占用点 {x},{y}（棋子数 {len(stones)}，我执 {my_color}）')
        return mv

    def new_game(self):
        """一局结束：END 收尾，下一局直接复用进程"""
        try:
            self._send('END')
        except RapfiError:
            pass

    def close(self):
        if self.p:
            try:
                self.p.kill()
            except Exception:
                pass
            self.p = None


# ── 对局执行器：带随机开局，避免确定性评估把同一盘棋重放 ─────────────
OPENING_POINTS = [(r, c) for r in range(5, 10) for c in range(5, 10)]   # 天元 5×5


def play_one_game(net_player, rapfi_player, net_black, first_move=None):
    """双方下完一局；first_move=(r,c) 指定黑1，None=不指定

    取代 Game.start_play：它会 init_board，无法注入随机开局。
    """
    from game import Board
    b = Board(width=15, height=15, n_in_row=5)
    b.init_board(0)
    net_color = 1 if net_black else 2
    net_player.set_player_ind(net_color)
    rapfi_player.set_player_ind(3 - net_color)
    if first_move is not None:
        b.do_move(first_move[0] * 15 + first_move[1])
    while True:
        end, winner = b.game_end()
        if end:
            return winner, len(b.states)
        if b.current_player == net_color:
            mv = net_player.get_action(b, temp=1e-3)
        else:
            mv = rapfi_player.get_action(b)
        if mv not in b.availables:
            raise RapfiError('非法落子 %r' % (mv,))
        b.do_move(mv)
