"""
Check the wiring on the Raspberry Pi WITHOUT the camera or the AI model.

Lights each status LED in turn, then runs through every brightness level of the LED light, then every speed of the
fan, then switches both off. Run it after wiring and before detect.py --hardware.

    python test_hardware.py
"""

import time

import config
from appliances import Appliances
from status_leds import COLOURS, StatusLeds


def step(message, seconds=3):
    print(message)
    time.sleep(seconds)


def main():
    print(f"Light: relay GPIO{config.LIGHT_RELAY_PIN}, MOSFET GPIO{config.LIGHT_PWM_PIN}\n"
          f"Fan:   relay GPIO{config.FAN_RELAY_PIN}, MOSFET GPIO{config.FAN_PWM_PIN}\n")
    devices = Appliances(use_hardware=True)
    leds = StatusLeds(devices, use_hardware=True)
    try:
        print("Status LEDs, one at a time:")
        for device in config.STATUS_LEDS:
            for colour in COLOURS:
                leds.leds[(device, colour)].on()
                step(f"  {device} {colour} (GPIO{config.STATUS_LEDS[device][colour]})", 1)
                leds.leds[(device, colour)].off()
        print("Start pattern: all six on for 3 s")
        leds._startup()
        leds._set(leds._all(), 0)
        print()
        for device, seconds in ((devices.light, 2), (devices.fan, 4)):
            device.level = 0
            device.set_power(True)
            step(f"{device.name} should be ON at {device.level_text()} (relay clicks)", seconds)
            for _ in range(len(device.levels) - 1):
                device.step_level(+1)
                step(f"{device.name} at {device.level_text()}", seconds)
            device.set_power(False)
            step(f"{device.name} should be OFF", 2)
        print("\nDone. If a device was on when it should be off, set "
              "RELAY_ACTIVE_HIGH in config.py the other way.")
    except KeyboardInterrupt:
        pass
    finally:
        devices.close()
        print("Stop pattern: flash 3x, then red for 1 s")
        leds.shutdown()
        leds.close()


if __name__ == "__main__":
    main()
