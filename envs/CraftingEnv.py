import gymnasium as gym
import numpy as np
from math import ceil
from typing import NamedTuple
from gymnasium import spaces

MAX_PROG = 50
MAX_QUAL = 100
MAX_DUR = 80
MAX_CP = 1000

ILLEGAL_PENALTY = 5

SEED = 0

ACTION_LIST = [
    "Basic Touch",
    "Standard Touch",
    "Advanced Touch",
    "Trained Eye",
    "Prudent Touch",
    "Preparatory Touch",
    "Focused Touch",
    "Byregot's",
    "Precise Touch",
    "Hasty Touch",
    "Delicate Synth",
    "Basic Synth",
    "Careful Synth",
    "Focused Synth",
    "Groundwork",
    "Rapid Synth",
    "Intensive Synth",
    "Prudent Synth",
    "Waste Not I",
    "Waste Not II",
    "Manipulation",
    "Muscle Memory",
    "Innovation",
    "Great Strides",
    "Veneration",
    "Tricks of the Trade",
    "Careful Observation",
    "Observe"
]

STATUSES = [
    "NORMAL",
    "GOOD",
    "EXCELLENT",
    "POOR",
    "CENTERED",
    "STURDY",
    "PLIANT",
    "MALLEABLE",
    "PRIMED"
]


class Recipe(NamedTuple):
    max_dur: int
    prog: float
    qual: float
    cp: int
    specialist: bool
    opener: str


class CraftingEnv(gym.Env):
    metadata = {"render.modes": ["ansi"]}

    def __init__(self, recipe, render_mode="ansi"):
        self._recipe = recipe
        self.render_mode = render_mode
        self.observation_space = spaces.Dict(
            {
                "prog_remaining": spaces.Box(0, MAX_PROG, dtype=np.float16),
                "qual_remaining": spaces.Box(0, MAX_QUAL, dtype=np.float16),
                "durability": spaces.Discrete(MAX_DUR+1),
                "cp": spaces.Discrete(MAX_CP+1),
                "innovation": spaces.Discrete(6+1),
                "great_strides": spaces.Discrete(5+1),
                "waste_not": spaces.Discrete(10+1),
                "manipulation": spaces.Discrete(10+1),
                "heart_and_soul": spaces.Discrete(1+1),
                "careful_observation": spaces.Discrete(3+1),
                "muscle_memory": spaces.Discrete(5+1),
                "veneration": spaces.Discrete(7+1),
                "inner_quiet": spaces.Discrete(10+1),
                "last_action": spaces.Discrete(len(ACTION_LIST)+1),
                "two_actions_ago_basic": spaces.Discrete(1+1),
                "status": spaces.MultiBinary(len(STATUSES))
            }
        )

        self._max_dur = recipe.max_dur
        self._prog = recipe.prog
        self._qual = recipe.qual
        self._dur = recipe.max_dur
        self._cp = recipe.cp
        self._inno = 0
        self._gs = 0
        self._wn = 0
        self._manip = 0
        self._has = 1 if recipe.specialist else 0
        self._co = 3 if recipe.specialist else 0
        self._mm = 0
        self._iq = 0
        self._ven = 0
        self._last_action = len(ACTION_LIST)  # deny combos
        self._two_actions_ago_basic = False  # I hate this goddamn state variable
        self.action_space = spaces.Discrete(len(ACTION_LIST))
        self.craft_complete = False
        self._status = self.rng_status()  # NORMAL
        self.success_mod = 0

        if recipe.opener == "MM":
            self.apply_prog(3, 10, 6)
            self._mm = 5
        elif recipe.opener == "Ref":
            self.apply_qual(1, 10, 18)
            self.iq = 2

    def apply_action(self, dur, cp):
        if self._wn:
            dur /= 2
        if self._status == 5:
            dur /= 2
        dur = ceil(dur)
        if self._status == 6:
            cp /= 2
            cp = ceil(cp)
        if self._cp < cp:
            return False
        if self._dur < ceil(dur):
            self.craft_complete = True
        self._dur -= dur
        if self._dur > self._max_dur:
            self._dur = self._max_dur
        self._cp -= cp
        return True

    def apply_qual(self, base, dur, cp):
        if not self.apply_action(dur, cp):
            return -ILLEGAL_PENALTY
        mul = (1 + self._iq / 10) * (1 + (0.5 if self._inno else 0) + (1 if self._gs else 0))

        if self._status == 1:
            mul *= 1.5
        elif self._status == 2:
            mul *= 4

        if self._gs:
            self._gs = 0
        if base * mul > self._qual:
            base = self._qual
            mul = 1
        self._qual -= base * mul
        self._iq += 1
        if self._iq > 10:
            self._iq = 10
        return base * mul

    def apply_prog(self, base, dur, cp):
        if not self.apply_action(dur, cp):
            return -ILLEGAL_PENALTY
        mul = (1 + (0.5 if self._ven else 0) + (1 if self._mm else 0))

        if self._status == 7:
            mul *= 1.5

        if self._mm:
            self._mm = 0
        if base * mul > self._prog:
            base = self._prog
            mul = 1
        self._prog -= base * mul
        return base * mul

    def tick_statuses(self):
        if self._mm:
            self._mm -= 1
        if self._gs:
            self._gs -= 1
        if self._inno:
            self._inno -= 1
        if self._ven:
            self._ven -= 1
        if self._wn:
            self._wn -= 1
        if self._manip:
            self._manip -= 1
            self._dur += 5
            if self._dur >= self._max_dur:
                self._dur = self._max_dur

    def step(self, action):
        reward = -1
        legal = True
        success_rate_bonus = (0.25 if self._status == 4 else 0) + self.success_mod
        status_bonus = 2 if self._status == 8 else 0

        if action == 0:  # Basic Touch
            reward += self.apply_qual(1, 10, 18)
            self._two_actions_ago_basic = False
        elif action == 1:  # Standard Touch
            if self._last_action == 0:
                self._two_actions_ago_basic = True
            reward += self.apply_qual(1.25, 10, 18 if self._last_action == 0 else 32)
        elif action == 2:  # Advanced Touch
            reward += self.apply_qual(1.5, 10, 18 if self._last_action == 1 and self._two_actions_ago_basic else 46)
            self._two_actions_ago_basic = False
        elif action == 3:  # Trained Eye
            if self._iq != 10:
                legal = False
            else:
                reward += self.apply_qual(1, 0, 32)
        elif action == 4:  # Prudent Touch
            if self._wn:
                legal = False
            else:
                reward += self.apply_qual(1, 5, 25)
        elif action == 5:  # Preparatory Touch
            reward += self.apply_qual(2, 20, 40)
            self._iq += 1
            if self._iq > 10:
                self._iq = 10
        elif action == 6:  # Focused Touch
            if self._last_action == 27 or \
                    self.np_random.random() < 0.5 + success_rate_bonus:  # Observe
                reward += self.apply_qual(1.5, 10, 18)
            else:
                reward += self.apply_qual(0, 10, 18)
        elif action == 7:  # Byregot's
            reward += self.apply_qual(1 + self._iq * 0.2, 10, 24)
            self._iq = 0
        elif action == 8:  # Precise
            if self._status == 1 or self._status == 2:
                # Allowed
                reward += self.apply_qual(1.5, 10, 18)
                self._iq += 1
                if self._iq > 10:
                    self._iq = 10
            elif self._has:
                self._has = 0
                reward += self.apply_qual(1.5, 10, 18)
                self._iq += 1
                if self._iq > 10:
                    self._iq = 10
            else:
                legal = False
        elif action == 9:  # Hasty
            if self.np_random.random() < 0.6 + success_rate_bonus:
                reward += self.apply_qual(1, 10, 0)
            else:
                reward += self.apply_qual(0, 10, 0)
        elif action == 10:  # Delicate o_o
            if (q := self.apply_qual(1, 10, 32)) != -ILLEGAL_PENALTY:
                reward += self.apply_prog(1, 0, 0) + q
            else:
                legal = False
        elif action == 11:  # Basic Synth
            reward += self.apply_prog(1.2, 10, 0)
        elif action == 12:  # Careful Synth
            reward += self.apply_prog(1.8, 10, 7)
        elif action == 13:  # Focused Synth
            if self._last_action == 27 or \
                    self.np_random.random() < 0.5 + success_rate_bonus:  # Observe
                reward += self.apply_prog(2, 10, 5)
            else:
                reward += self.apply_prog(0, 10, 5)
        elif action == 14:  # Groundwork
            reward += self.apply_prog(3.6, 20, 18)
        elif action == 15:  # Rapid Synth
            if self.np_random.random() < 0.6 + success_rate_bonus:
                reward += self.apply_prog(5, 10, 0)
            else:
                reward += self.apply_action(10, 0)
        elif action == 16:  # Intensive Synth
            if self._status == 1 or self._status == 2:
                # Allowed
                reward += self.apply_prog(4, 10, 6)
            elif self._has:
                self._has = 0
                reward += self.apply_qual(4, 10, 6)
            else:
                legal = False
        elif action == 17:  # Prudent Synth
            if self._wn:
                legal = False
            else:
                reward += self.apply_prog(1.8, 5, 18)
        elif action == 18:  # WN1
            if self.apply_action(0, 56):
                self._wn = 5 + status_bonus  # Set to 1 higher because it'll automatically tick down
            else:
                legal = False
        elif action == 19:  # WN2
            if self.apply_action(0, 98):
                self._wn = 9 + status_bonus  # Set to 1 higher because it'll automatically tick down
            else:
                legal = False
        elif action == 20:  # Manip
            if self.apply_action(0, 96):
                self._manip = 9 + status_bonus  # Set to 1 higher because it'll automatically tick down
                self._dur -= 5  # I love jank
            else:
                legal = False
        elif action == 21:  # MM
            if self.apply_action(0, 88):
                self._dur += 30
                if self._dur >= self._max_dur:
                    self._dur = self._max_dur
            else:
                legal = False
        elif action == 22:  # Inno
            if self.apply_action(0, 18):
                self._inno = 5 + status_bonus  # Set to 1 higher because it'll automatically tick down
            else:
                legal = False
        elif action == 23:  # GS
            if self.apply_action(0, 32):
                self._gs = 4 + status_bonus  # Set to 1 higher because it'll automatically tick down
            else:
                legal = False
        elif action == 24:  # ven
            if self.apply_action(0, 18):
                self._ven = 5 + status_bonus  # Set to 1 higher because it'll automatically tick down
            else:
                legal = False
        elif action == 25:  # Tricks
            if self._status == 1 or self._status == 2:
                # Allowed
                self._cp += 20
            elif self._has:
                self._has = 0
                self._cp += 20
            else:
                legal = False
        elif action == 26:  # CO
            if self._co:
                self._co -= 1
            else:
                legal = False
            pass
        elif action == 27:  # Observe
            if self.apply_action(0, 7):
                pass  # Do literally nothing
            else:
                legal = False

        if legal:  # A legal action was performed, and statuses should tick
            self.tick_statuses()
            if self._last_action == 0:
                self._two_actions_ago_basic = True
            else:
                self._two_actions_ago_basic = False
            self._last_action = action
            self._status = self.rng_status()
        else:
            reward -= ILLEGAL_PENALTY
            # State should be preserved

        if self.is_finished():
            if self._qual <= 0:
                reward += 10
            if self._prog <= 0:
                reward += 10
            else:
                reward -= 0
            if self._qual <= 0 and self._prog <= 0:
                reward += 100

        return self.get_obs(), reward, self.is_finished(), False, {}

    def is_finished(self):
        return self.craft_complete or self._cp < 0

    def rng_status(self):
        roll = self.np_random.random()

        if roll < 0.12:
            return STATUSES.index("PRIMED")
        elif roll < 0.12 + 0.12:
            return STATUSES.index("MALLEABLE")
        elif roll < 0.12 + 0.12 + 0.15:
            return STATUSES.index("STURDY")
        elif roll < 0.12 + 0.12 + 0.15 + 0.12:
            return STATUSES.index("PLIANT")
        elif roll < 0.12 + 0.12 + 0.15 + 0.12 + 0.15:
            return STATUSES.index("CENTERED")
        elif roll < 0.12 + 0.12 + 0.15 + 0.12 + 0.15 + 0.17:
            return STATUSES.index("GOOD")
        else:
            return 0

    def get_obs(self):
        status = [0 for i in range(len(STATUSES))]
        status[self._status] = 1
        return {
                "prog_remaining": self._prog,
                "qual_remaining": self._qual,
                "durability": self._dur,
                "cp": self._cp,
                "innovation": self._inno,
                "great_strides": self._gs,
                "waste_not": self._wn,
                "manipulation": self._manip,
                "heart_and_soul": self._has,
                "careful_observation": self._co,
                "muscle_memory": self._mm,
                "veneration": self._ven,
                "inner_quiet": self._iq,
                "last_action": self._last_action,
                "two_actions_ago_basic": self._two_actions_ago_basic,
                "status": status
            }

    def _load_obs(self, observation):
        if "prog_remaining" in observation:
            self._prog = observation["prog_remaining"]
        if "qual_remaining" in observation:
            self._qual = observation["qual_remaining"]
        if "durability" in observation:
            self._dur = observation["durability"]
        if "cp" in observation:
            self._cp = observation["cp"]
        if "innovation" in observation:
            self._inno = observation["innovation"]
        if "great_strides" in observation:
            self._gs = observation["great_strides"]
        if "waste_not" in observation:
            self._wn = observation["waste_not"]
        if "manipulation" in observation:
            self._manip = observation["manipulation"]
        if "heart_and_soul" in observation:
            self._has = observation["heart_and_soul"]
        if "careful_observation" in observation:
            self._co = observation["careful_observation"]
        if "muscle_memory" in observation:
            self._mm = observation["muscle_memory"]
        if "veneration" in observation:
            self._ven = observation["veneration"]
        if "inner_quiet" in observation:
            self._iq = observation["inner_quiet"]
        if "last_action" in observation:
            self._last_action = observation["last_action"]
        if "two_actions_ago_basic" in observation:
            self._two_actions_ago_basic = observation["two_actions_ago_basic"]
        if "status" in observation:
            self._status = observation["status"]

    def reset(self, seed=SEED, options={}):
        self.craft_complete = False
        self.__init__(self._recipe)
        if options and "initial_state" in options:
            self._load_obs(options["initial_state"])

        return self.get_obs(), None

    def render(self):
        if self.render_mode == "ansi":
            base = f"Prog left: {self._prog*226}\nQual left: {self._qual*262}\n" + \
                f"CP: {self._cp} Dur: {self._dur} IQ {self._iq}\n" + \
                f"Status: {STATUSES[self._status]}"
            statuses = "Statuses:\n"
            if self._inno > 0:
                statuses += f"Inno {self._inno} "
            if self._gs > 0:
                statuses += f"GS {self._gs} "
            if self._wn > 0:
                statuses += f"WN {self._wn} "
            if self._manip > 0:
                statuses += f"Manip {self._manip} "
            if self._ven > 0:
                statuses += f"Ven {self._ven} "
            if self._mm > 0:
                statuses += f"MM {self._mm} "
            statuses = statuses.strip()
            if statuses == "Statuses:":
                return base
            else:
                return base + "\n" + statuses

    def score(self):
        return self._prog + self._qual

    def set_status(self, status):
        self._status = status

    def get_status(self):
        return STATUSES[self._status]

    def mod_success(self, success):
        self.success_mod = success