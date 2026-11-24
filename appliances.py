"""
Appliance control: an LED light and an axial fan, each with power and level.

Each device has a relay that switches its POWER and a MOSFET driven with PWM
that sets its LEVEL (LED brightness / fan speed). Gestures act on whichever
device is selected ("one" = light, "peace" = fan).

The same class works in two modes:
  * simulated: nothing physical happens, the state is only drawn on screen.
  * hardware:  on a Raspberry Pi, GPIO pins drive the relays and MOSFETs.
"""

import math
import time

import cv2
import numpy as np

import config


class Device:
    """One appliance: a relay for power plus a PWM MOSFET for its level."""

    def __init__(self, name, levels, start_level, relay_pin, pwm_pin, pwm_frequency,
                 use_hardware):
        self.name = name
        self.levels = levels
        self.level = start_level      # index into levels, kept while off
        self.on = False
        self.relay = None
        self.pwm = None
        if use_hardware:
            # gpiozero picks the right backend on a Pi 5 (lgpio) by itself.
            from gpiozero import OutputDevice, PWMOutputDevice
            self.relay = OutputDevice(relay_pin, active_high=config.RELAY_ACTIVE_HIGH,
                                      initial_value=False)
            self.pwm = PWMOutputDevice(pwm_pin, frequency=pwm_frequency,
                                       initial_value=0.0)

    @property
    def duty(self):
        return self.levels[self.level] if self.on else 0.0

    def level_text(self):
        return f"{self.levels[self.level]:.0%}"

    def set_power(self, on):
        self.on = on
        self._apply()

    def step_level(self, delta):
        """Move one level up or down. Turns the device on if it was off."""
        if self.on:
            self.level = max(0, min(len(self.levels) - 1, self.level + delta))
        self.on = True
        self._apply()

    def _apply(self):
        if self.pwm is not None:
            self.pwm.value = self.duty
        if self.relay is not None:
            self.relay.value = self.on

    def close(self):
        self.set_power(False)
        for part in (self.pwm, self.relay):
            if part is not None:
                part.close()


class Appliances:
    def __init__(self, use_hardware=False):
        self.use_hardware = use_hardware
        self.light = Device("Light", config.LIGHT_LEVELS, config.LIGHT_START_LEVEL,
                            config.LIGHT_RELAY_PIN, config.LIGHT_PWM_PIN,
                            config.LIGHT_PWM_FREQUENCY, use_hardware)
        self.fan = Device("Fan", config.FAN_LEVELS, config.FAN_START_LEVEL,
                          config.FAN_RELAY_PIN, config.FAN_PWM_PIN,
                          config.FAN_PWM_FREQUENCY, use_hardware)
        self.selected = self.light
        self.history = []                 # recent actions, newest last
        self._fan_angle = 0.0

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def perform(self, action):
        """Run an action name from config.GESTURE_ACTIONS. Returns a message."""
        device = self.selected
        if action == "select_light":
            self.selected = self.light
            message = "Light selected"
        elif action == "select_fan":
            self.selected = self.fan
            message = "Fan selected"
        elif action == "power_toggle":
            device.set_power(not device.on)
            message = (f"{device.name} ON ({device.level_text()})" if device.on
                       else f"{device.name} OFF")
        elif action == "level_up":
            device.step_level(+1)
            message = f"{device.name} {device.level_text()}"
        elif action == "level_down":
            device.step_level(-1)
            message = f"{device.name} {device.level_text()}"
        elif action == "all_off":
            self.light.set_power(False)
            self.fan.set_power(False)
            message = "Everything OFF"
        else:
            return f"Unknown action: {action}"

        self.history.append((time.strftime("%H:%M:%S"), message))
        self.history = self.history[-5:]
        return message

    def close(self):
        """Switch everything off and free the GPIO pins."""
        self.light.close()
        self.fan.close()

    # ------------------------------------------------------------------
    # On-screen dashboard
    # ------------------------------------------------------------------
    def draw_dashboard(self, height, width=300):
        """Return an image (BGR) showing the state of every appliance."""
        panel = np.full((height, width, 3), 40, dtype=np.uint8)
        white = (255, 255, 255)
        grey = (110, 110, 110)
        select = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        mode = "HARDWARE" if self.use_hardware else "SIMULATED"
        cv2.putText(panel, f"Appliances ({mode})", (10, 25), font, 0.55, white, 1)

        # Light: a bulb whose glow follows the brightness.
        light = self.light
        if light is self.selected:
            cv2.rectangle(panel, (4, 38), (width - 4, 150), select, 2)
        cx, cy = 60, 85
        if light.on:
            glow = tuple(int(c * (0.35 + 0.65 * light.duty)) for c in (0, 220, 255))
            cv2.circle(panel, (cx, cy), 38, glow, -1)
            cv2.circle(panel, (cx, cy), 46, glow, 2)
        else:
            cv2.circle(panel, (cx, cy), 38, grey, 2)
        cv2.rectangle(panel, (cx - 14, cy + 38), (cx + 14, cy + 52), grey, -1)
        cv2.putText(panel, "LIGHT", (125, 75), font, 0.6, white, 1)
        cv2.putText(panel, f"ON {light.level_text()}" if light.on else "OFF", (125, 105),
                    font, 0.8, (0, 220, 255) if light.on else grey, 2)

        # Fan: blades that spin faster at higher speed.
        fan = self.fan
        if fan is self.selected:
            cv2.rectangle(panel, (4, 158), (width - 4, 265), select, 2)
        cx, cy = 60, 210
        self._fan_angle = (self._fan_angle + 30 * fan.duty) % 360
        colour = (255, 200, 0) if fan.on else grey
        cv2.circle(panel, (cx, cy), 45, colour, 2)
        for blade in range(3):
            angle = math.radians(self._fan_angle + blade * 120)
            tip = (int(cx + 38 * math.cos(angle)), int(cy + 38 * math.sin(angle)))
            cv2.line(panel, (cx, cy), tip, colour, 6)
        cv2.circle(panel, (cx, cy), 7, white, -1)
        cv2.putText(panel, "FAN", (125, 200), font, 0.6, white, 1)
        cv2.putText(panel, f"ON {fan.level_text()}" if fan.on else "OFF", (125, 230),
                    font, 0.8, colour, 2)

        cv2.putText(panel, f"Selected: {self.selected.name.upper()}", (10, 295),
                    font, 0.6, select, 2)

        # Last few actions.
        y = 335
        cv2.putText(panel, "Recent actions:", (10, y), font, 0.5, white, 1)
        for stamp, message in reversed(self.history[-4:]):
            y += 22
            if y > height - 5:
                break
            cv2.putText(panel, f"{stamp}  {message}", (10, y), font, 0.45,
                        (180, 220, 180), 1)
        return panel
