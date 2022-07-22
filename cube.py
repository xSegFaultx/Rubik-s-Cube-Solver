import numpy as np
import copy
import collections


class Cube:
    def __init__(self):
        """
        cube structure: [Up, Front, Down, Left, Right, Back]

        int code for each side:
        Left:0
        Front: 1
        Right: 2
        Back: 3
        Up: 4
        Down: 5
        """
        self.base = np.ones((3, 3))
        self.cube = np.array([self.base * i for i in range(6)], dtype=int)
        self.int2face = ['L', 'F', 'R', 'B', 'U', 'D']
        origin_string = "LFRBUD"
        translated_string = "012345"
        self.string2int = str.maketrans(origin_string, translated_string)
        self.action_param = [[0, True],
                             [0, False],
                             [2, True],
                             [2, False], ]
        self.state_structure = collections.namedtuple("State",
                                                      field_names=['corner_pos', 'side_pos', 'corner_ort', 'side_ort'])
        self.simple_state = self.state_structure(corner_pos=tuple(range(8)), side_pos=tuple(range(12)),
                                                 corner_ort=tuple([0] * 8),
                                                 side_ort=tuple([0] * 12))
        self.transform_map = [
            [
                ((0, 3), (1, 0), (2, 1), (3, 2)),
                ((0, 3), (1, 0), (2, 1), (3, 2)),
                (),
                ()
            ],
            [
                ((0, 3), (1, 0), (2, 1), (3, 2)),
                ((0, 3), (1, 0), (2, 1), (3, 2)),
                (),
                ()
            ],
            [
                ((4, 5), (5, 6), (6, 7), (7, 4)),
                ((8, 9), (9, 10), (10, 11), (11, 8)),
                (),
                ()
            ],
            [
                ((4, 5), (5, 6), (6, 7), (7, 4)),
                ((8, 9), (9, 10), (10, 11), (11, 8)),
                (),
                ()
            ],
            [
                ((3, 0), (7, 3), (0, 4), (4, 7)),
                ((7, 3), (3, 4), (11, 7), (4, 11)),
                ((0, 1), (3, 2), (4, 2), (7, 1)),
                ()
            ],
            [
                ((3, 0), (7, 3), (0, 4), (4, 7)),
                ((7, 3), (3, 4), (11, 7), (4, 11)),
                ((0, 1), (3, 2), (4, 2), (7, 1)),
                ()
            ],
            [
                ((1, 2), (2, 6), (6, 5), (5, 1)),
                ((1, 6), (6, 9), (9, 5), (5, 1)),
                ((1, 2), (2, 1), (5, 1), (6, 2)),
                ()
            ],
            [
                ((1, 2), (2, 6), (6, 5), (5, 1)),
                ((1, 6), (6, 9), (9, 5), (5, 1)),
                ((1, 2), (2, 1), (5, 1), (6, 2)),
                ()
            ],
            [
                ((0, 1), (1, 5), (5, 4), (4, 0)),
                ((0, 5), (4, 0), (5, 8), (8, 4)),
                ((0, 2), (1, 1), (4, 1), (5, 2)),
                (0, 4, 5, 8)
            ],
            [
                ((0, 1), (1, 5), (5, 4), (4, 0)),
                ((0, 5), (4, 0), (5, 8), (8, 4)),
                ((0, 2), (1, 1), (4, 1), (5, 2)),
                (0, 4, 5, 8)
            ],
            [
                ((2, 3), (3, 7), (7, 6), (6, 2)),
                ((2, 7), (6, 2), (7, 10), (10, 6)),
                ((2, 2), (3, 1), (6, 1), (7, 2)),
                (2, 6, 7, 10)
            ],
            [
                ((2, 3), (3, 7), (7, 6), (6, 2)),
                ((2, 7), (6, 2), (7, 10), (10, 6)),
                ((2, 2), (3, 1), (6, 1), (7, 2)),
                (2, 6, 7, 10)
            ]
        ]
        self.count = 0

    def act(self, action, peek=False, simple=False):
        """
        action 0: turn the top axis z slice left 90 degree
        action 1: turn the top axis z slice right 90 degree
        action 2: turn the bottom axis z slice left 90 degree
        action 3: turn the bottom axis z slice right 90 degree

        action 4: turn the left axis y slice up 90 degree
        action 5: turn the left axis y slice down 90 degree
        action 6: turn the right axis y slice up 90 degree
        action 7: turn the right axis y slice down 90 degree

        action 8: turn the front axis x slice clockwise 90 degree
        action 9: turn the front axis x slice counter-clockwise 90 degree
        action 10: turn the back axis x slice clockwise 90 degree
        action 11: turn the back axis x slice counter-clockwise 90 degree
        """
        level, direction = self.action_param[action % 4]
        if 4 > action >= 0:
            self._z_spin(level, direction)
        elif 8 > action >= 4:
            self._y_spin(level, direction)
        elif 12 > action >= 8:
            self._x_spin(level, direction)
        if not simple:
            if peek:
                return self.transform(action)
            else:
                self.simple_state = self.transform(action)

    def step(self, action):
        curren_state = self.get_state()
        self.act(action)
        next_state = self.get_state()
        reward = -1
        terminal_state = False
        solved = False
        if self.solved():
            reward = 1
            terminal_state = True
            solved = True
        self.count += 1
        if self.count >= 50:
            terminal_state = True
        return curren_state, reward, terminal_state, next_state, solved

    def step_simple(self, action):
        curren_state = self.to_string()
        self.act(action, simple=True)
        next_state = self.to_string()
        if self.solved():
            reward = 1
            terminal_state = True
        else:
            reward = -1
            terminal_state = False
        return curren_state, next_state, reward, terminal_state

    def peek(self, target_action):
        current_string_state = self.to_string()
        curren_state = self.get_state()
        rewards = []
        next_states = []
        terminal_states = []
        for action in range(12):
            self.from_string(current_string_state, reset_count=False)
            temp_state = self.act(action, peek=True)
            next_state = self._get_state(temp_state)
            reward = -1
            terminal_state = False
            if self.solved():
                reward = 1
                terminal_state = True
            if self.count+1>50:
                terminal_state = True
            rewards.append(reward)
            next_states.append(next_state)
            terminal_states.append(terminal_state)
        self.from_string(current_string_state, reset_count=False)
        _, target_reward, target_terminal, target_next, target_solved = self.step(target_action)
        return curren_state, rewards, terminal_states, next_states, target_reward, target_terminal, target_next, target_solved

    def _z_spin(self, level, left_turn):
        """
        level 0: top axis z slice
        level 2: bottom axis z slice
        """
        if level == 0:
            if left_turn:
                self.cube[4] = np.rot90(self.cube[4], k=3)
            else:
                self.cube[4] = np.rot90(self.cube[4], k=1)
        elif level == 2:
            if left_turn:
                self.cube[5] = np.rot90(self.cube[5], k=1)
            else:
                self.cube[5] = np.rot90(self.cube[5], k=3)
        z_section = np.hstack((self.cube[0], self.cube[1], self.cube[2], self.cube[3]))
        selected_slice = z_section[level]
        if left_turn:
            result_slice = np.concatenate((selected_slice, selected_slice[0:3]))[3:]
        else:
            result_slice = np.concatenate((selected_slice[-3:], selected_slice))[:-3]
        z_section[level] = result_slice
        self.cube[0] = z_section[:, :3]
        self.cube[1] = z_section[:, 3:6]
        self.cube[2] = z_section[:, 6:9]
        self.cube[3] = z_section[:, 9:]

    def _y_spin(self, level, up_turn):
        """
        level 0: left axis z slice
        level 2: right axis z slice
        """
        if level == 0:
            if up_turn:
                self.cube[0] = np.rot90(self.cube[0], k=1)
            else:
                self.cube[0] = np.rot90(self.cube[0], k=3)
        elif level == 2:
            if up_turn:
                self.cube[2] = np.rot90(self.cube[2], k=3)
            else:
                self.cube[2] = np.rot90(self.cube[2], k=1)
        y_section = np.vstack((self.cube[4], self.cube[1], self.cube[5], np.rot90(self.cube[3], k=2)))
        y_section = y_section.T
        selected_slice = y_section[level]
        if up_turn:
            result_slice = np.concatenate((selected_slice, selected_slice[0:3]))[3:]
        else:
            result_slice = np.concatenate((selected_slice[-3:], selected_slice))[:-3]
        y_section[level] = result_slice
        y_section = y_section.T
        self.cube[4] = y_section[:3, :]
        self.cube[1] = y_section[3:6, :]
        self.cube[5] = y_section[6:9, :]
        self.cube[3] = np.rot90(y_section[9:, :], k=2)

    def _x_spin(self, level, clockwise):
        """
        level 0: front axis x slice
        level 1: middle axis x slice
        level 2: back axis x slice
        """
        if level == 0:
            if clockwise:
                self.cube[1] = np.rot90(self.cube[1], k=3)
            else:
                self.cube[1] = np.rot90(self.cube[1], k=1)
        elif level == 2:
            if clockwise:
                self.cube[3] = np.rot90(self.cube[3], k=1)
            else:
                self.cube[3] = np.rot90(self.cube[3], k=3)
        transformed_top = np.rot90(self.cube[4], k=3)
        transformed_bottom = np.rot90(self.cube[5], k=1)
        transformed_left = np.rot90(self.cube[0], k=2)
        x_section = np.vstack((transformed_top, self.cube[2], transformed_bottom, transformed_left))
        x_section = x_section.T
        selected_slice = x_section[level]
        if not clockwise:
            result_slice = np.concatenate((selected_slice, selected_slice[0:3]))[3:]
        else:
            result_slice = np.concatenate((selected_slice[-3:], selected_slice))[:-3]
        x_section[level] = result_slice
        x_section = x_section.T
        self.cube[4] = np.rot90(x_section[:3, :], k=1)
        self.cube[2] = x_section[3:6, :]
        self.cube[5] = np.rot90(x_section[6:9, :], k=3)
        self.cube[0] = np.rot90(x_section[9:, :], k=2)

    def scramble(self, steps=25):
        actions = np.random.randint(12, size=steps)
        for action in actions:
            self.act(action)

    def solved(self):
        for face in self.cube:
            center = face[1][1]
            flattened = np.ravel(face)
            face_match = np.all(flattened == center)
            if not face_match:
                return False
        return True

    def reset(self):
        self.cube = np.array([self.base * i for i in range(6)], dtype=int)
        self.simple_state = self.state_structure(corner_pos=tuple(range(8)), side_pos=tuple(range(12)),
                                                 corner_ort=tuple([0] * 8),
                                                 side_ort=tuple([0] * 12))
        self.count = 0

    def from_string(self, s, reset_count=True):
        translated = s.translate(self.string2int)
        translated = " ".join(list(translated))
        state = np.fromstring(translated, dtype=int, sep=" ").reshape((6, 3, 3))
        self.cube[0] = state[4]
        self.cube[1] = state[2]
        self.cube[2] = state[1]
        self.cube[3] = state[5]
        self.cube[4] = state[0]
        self.cube[5] = state[3]
        if reset_count:
            self.count = 0

    def to_string(self):
        flattened = np.array(
            [self.cube[4], self.cube[2], self.cube[1], self.cube[5], self.cube[0], self.cube[3]]).flatten()
        string_cube = []
        for c in flattened:
            string_cube.append(self.int2face[c])
        return "".join(string_cube)

    def visualize(self):
        padding = [["*", "*", "*"],
                   ["*", "*", "*"],
                   ["*", "*", "*"]]
        top = copy.deepcopy(padding)
        left = copy.deepcopy(padding)
        front = copy.deepcopy(padding)
        right = copy.deepcopy(padding)
        back = copy.deepcopy(padding)
        bottom = copy.deepcopy(padding)
        for i in range(3):
            for j in range(3):
                top[i][j] = self.int2face[self.cube[4][i][j]]
                left[i][j] = self.int2face[self.cube[0][i][j]]
                front[i][j] = self.int2face[self.cube[1][i][j]]
                right[i][j] = self.int2face[self.cube[2][i][j]]
                back[i][j] = self.int2face[self.cube[3][i][j]]
                bottom[i][j] = self.int2face[self.cube[5][i][j]]
        top_vis = np.hstack((padding, top, padding, padding))
        mid_vis = np.hstack((left, front, right, back))
        bottom_vis = np.hstack((padding, bottom, padding, padding))
        for i in range(top_vis.shape[0]):
            print("   ".join(top_vis[i]))
        for i in range(mid_vis.shape[0]):
            print("   ".join(mid_vis[i]))
        for i in range(bottom_vis.shape[0]):
            print("   ".join(bottom_vis[i]))

    def transform(self, action):
        if action in [7, 4, 1, 2, 9, 10]:
            is_inv = True
        else:
            is_inv = False
        c_map, s_map, c_rot, s_flp = self.transform_map[action]
        corner_pos = _permute(self.simple_state.corner_pos, c_map, is_inv)
        corner_ort = _permute(self.simple_state.corner_ort, c_map, is_inv)
        corner_ort = _rotate(corner_ort, c_rot)
        side_pos = _permute(self.simple_state.side_pos, s_map, is_inv)
        side_ort = self.simple_state.side_ort
        if s_flp:
            side_ort = _permute(side_ort, s_map, is_inv)
            side_ort = _flip(side_ort, s_flp)
        return self.state_structure(corner_pos=tuple(corner_pos), corner_ort=tuple(corner_ort),
                                    side_pos=tuple(side_pos), side_ort=tuple(side_ort))

    def get_state(self):
        encode = np.zeros((20, 24))
        for corner_idx in range(8):
            perm_pos = self.simple_state.corner_pos.index(corner_idx)
            corn_ort = self.simple_state.corner_ort[perm_pos]
            encode[corner_idx, perm_pos * 3 + corn_ort] = 1

        for side_idx in range(12):
            perm_pos = self.simple_state.side_pos.index(side_idx)
            side_ort = self.simple_state.side_ort[perm_pos]
            encode[8 + side_idx, perm_pos * 2 + side_ort] = 1
        return encode

    def _get_state(self, state):
        encode = np.zeros((20, 24))
        for corner_idx in range(8):
            perm_pos = state.corner_pos.index(corner_idx)
            corn_ort = state.corner_ort[perm_pos]
            encode[corner_idx, perm_pos * 3 + corn_ort] = 1

        for side_idx in range(12):
            perm_pos = state.side_pos.index(side_idx)
            side_ort = state.side_ort[perm_pos]
            encode[8 + side_idx, perm_pos * 2 + side_ort] = 1
        return encode


# below are code for simple cube state
def _permute(t, m, is_inv=False):
    r = list(t)
    for from_idx, to_idx in m:
        if is_inv:
            r[from_idx] = t[to_idx]
        else:
            r[to_idx] = t[from_idx]
    return r


def _rotate(corner_ort, corners):
    r = list(corner_ort)
    for c, angle in corners:
        r[c] = (r[c] + angle) % 3
    return r


def _map_orient(cols, orient_id):
    if orient_id == 0:
        return cols
    elif orient_id == 1:
        return cols[2], cols[0], cols[1]
    else:
        return cols[1], cols[2], cols[0]


def _flip(side_ort, sides):
    return [
        o if idx not in sides else 1 - o
        for idx, o in enumerate(side_ort)
    ]
