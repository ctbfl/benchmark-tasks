"""Task clocks, physical key input, and the binary temperature score."""
import json
from decimal import Decimal
from pathlib import Path

from electronics import (TimerState, CalculatorState, pressed_buttons,
                         update_timer_display, update_calculator_display)
from lab_glass import update_temperature
from scene import CONFIG, ROOT

HERE = Path(__file__).resolve().parent


class TaskRuntime:
    def __init__(self, env):
        self.base = env.unwrapped
        self.timer = TimerState()
        self.calculator = CalculatorState()
        self.time = 0.0
        self.temperature = 35.0
        self.pressed = {"timer": set(), "calculator": set()}
        metadata = json.loads((HERE / "assets/electronics/metadata.json").read_text())
        self.key_ids = {}
        self.thresholds = {}
        for name in self.pressed:
            obj = self.base.scene[name]
            self.key_ids[name] = {key: obj.joint_names.index(item["joint"])
                                 for key, item in metadata[name]["buttons"].items()}
            self.thresholds[name] = {"press_threshold": metadata[name]["press_threshold_m"],
                                     "release_threshold": metadata[name]["release_threshold_m"]}
        self.refresh()

    def refresh(self):
        stage = self.base.sim.stage
        update_timer_display(stage, ROOT + "/timer", self.timer)
        update_calculator_display(stage, ROOT + "/calculator", self.calculator)
        update_temperature(stage, ROOT + "/thermometer", self.temperature)

    def tick(self, dt):
        for name, state in (("timer", self.timer), ("calculator", self.calculator)):
            values = self.base.scene[name].data.joint_pos.torch[0].cpu().numpy()
            positions = {key: float(values[index]) for key, index in self.key_ids[name].items()}
            self.pressed[name] = pressed_buttons(positions, self.pressed[name], **self.thresholds[name])
            state.update_buttons(self.pressed[name])
        self.timer.advance(dt)
        self.time += dt
        self.temperature = 35 + 10 * min(self.time / 15, 1)

    def step(self, action, on_physics_step=None):
        """Use the unmodified native 14-D YAM action manager at every substep."""
        base = self.base
        dt = 1 / CONFIG["simulation"]["physics_hz"]
        base.action_manager.process_action(action)
        for _ in range(CONFIG["simulation"]["physics_hz"] // CONFIG["simulation"]["control_hz"]):
            base.action_manager.apply_action()
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(dt)
            self.tick(dt)
            if on_physics_step is not None:
                on_physics_step()
        self.refresh()

    @property
    def score(self):
        return int(self.calculator.last_submission == Decimal(45))

    def state(self):
        result = self.calculator.last_submission
        return {"time_s": self.time, "temperature_C": self.temperature,
                "timer": self.timer.display, "timer_running": self.timer.running,
                "calculator": self.calculator.display,
                "submitted": None if result is None else str(result), "score": self.score}
