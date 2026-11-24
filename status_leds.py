"""
Six status LEDs (two trios of green / yellow / red) that show what the system
is doing, so you can follow it without looking at a screen.

Normal display (between animations), one trio per device:
    red    = device OFF
    green  = device ON, brighter at higher brightness / speed
    yellow = device is SELECTED (palm, like and dislike act on it)

Animations (each about one second, then back to the normal display):
    system start  all six ON for 3 s
    system stop   all six flash 3x, then both reds ON for 1 s, then dark
    one / peace   that device's yellow blinks 3x (selected)
    palm -> ON    trio sweeps red -> yellow -> green
    palm -> OFF   trio sweeps green -> yellow -> red
    like          trio chases green -> yellow -> red twice (going up)
    dislike       trio chases red -> yellow -> green twice (going down)
    at the limit  like at max / dislike at min: yellow double-flash
    three         both reds flash 3x (everything off)

Animations run on a background thread, so gesture detection never waits.
Without --hardware this class does nothing.
"""

import queue
import threading
import time

import config

COLOURS = ("green", "yellow", "red")


class StatusLeds:
    def __init__(self, appliances, use_hardware=False):
        self.appliances = appliances
        self.enabled = use_hardware
        self.leds = {}
        if not use_hardware:
            return
        from gpiozero import PWMLED
        for device, pins in config.STATUS_LEDS.items():
            for colour in COLOURS:
                self.leds[(device, colour)] = PWMLED(pins[colour], frequency=200)
        self.jobs = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------
    # Public calls (all return immediately unless wait=True)
    # ------------------------------------------------------------------
    def startup(self):
        self._add(self._startup)

    def shutdown(self):
        """Play the stop pattern and wait for it to finish (about 2.5 s)."""
        if not self.enabled:
            return
        done = threading.Event()
        self._add(self._shutdown, done)
        done.wait(timeout=5)

    def gesture(self, gesture, device, was_on, was_level):
        """Call after appliances.perform(). device = the device it acted on."""
        name = device.name
        if gesture in ("one", "peace"):
            self._add(lambda: self._blink([(name, "yellow")], 3, 0.12))
        elif gesture == "palm":
            order = COLOURS[::-1] if device.on else COLOURS   # ON ends on green
            self._add(lambda: self._sweep(name, order, 1))
        elif gesture in ("like", "dislike"):
            if was_on and device.level == was_level:          # already at the limit
                self._add(lambda: self._blink([(name, "yellow")], 2, 0.08))
            else:
                order = COLOURS if gesture == "like" else COLOURS[::-1]
                self._add(lambda: self._sweep(name, order, 2))
        elif gesture == "three":
            self._add(lambda: self._blink(self._all("red"), 3, 0.15))

    def close(self):
        if self.enabled:
            for led in self.leds.values():
                led.close()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------
    def _add(self, job, done=None):
        if self.enabled:
            self.jobs.put((job, done))

    def _run(self):
        while True:
            job, done = self.jobs.get()
            try:
                job()
                if done is None:
                    self._show_state()
            except Exception as error:   # never let an LED problem stop detection
                print(f"Status LED error: {error}")
            if done is not None:
                done.set()

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------
    def _set(self, keys, value):
        for key in keys:
            self.leds[key].value = value

    def _all(self, colour=None):
        return [k for k in self.leds if colour is None or k[1] == colour]

    def _blink(self, keys, times, seconds):
        for _ in range(times):
            self._set(self._all(), 0)
            self._set(keys, 1)
            time.sleep(seconds)
            self._set(keys, 0)
            time.sleep(seconds)

    def _sweep(self, device, order, times):
        self._set(self._all(), 0)
        for _ in range(times):
            for colour in order:
                self._set([(device, colour)], 1)
                time.sleep(0.15)
                self._set([(device, colour)], 0)
        self._set([(device, order[-1])], 1)
        time.sleep(0.25)

    def _startup(self):
        self._set(self._all(), 1)
        time.sleep(3)

    def _shutdown(self):
        for _ in range(3):
            self._set(self._all(), 1)
            time.sleep(0.25)
            self._set(self._all(), 0)
            time.sleep(0.25)
        self._set(self._all("red"), 1)
        time.sleep(1)
        self._set(self._all(), 0)

    def _show_state(self):
        apps = self.appliances
        for device in (apps.light, apps.fan):
            name = device.name
            if device.on:
                steps = len(device.levels) - 1
                fraction = device.level / steps if steps else 1.0
                low = config.STATUS_LED_MIN_GLOW
                self.leds[(name, "green")].value = low + (1 - low) * fraction
                self.leds[(name, "red")].value = 0
            else:
                self.leds[(name, "green")].value = 0
                self.leds[(name, "red")].value = 1
            self.leds[(name, "yellow")].value = 1 if device is apps.selected else 0
