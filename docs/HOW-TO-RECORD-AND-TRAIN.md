# Record new photos, train two CNNs, compare them

Everything runs in PowerShell on the laptop, from the `Industrial Project` folder.
Your old files are saved in `gesture_prototype\backup_20261106\`, and your old
photos stay in `dataset\` (new photos are added next to them).

## 1. Record 3 new sessions (about 5 minutes each)

```powershell
cd "C:\Users\haile\Downloads\Industrial Project\gesture_prototype"
..\laptop-venv\Scripts\python.exe collect_data.py
```

For each gesture, put your hand in the green box, press **SPACE**, and keep
the gesture while you slowly move your hand around the box (closer, further,
a little tilted). It saves 60 photos per gesture, one every 0.4 s, then moves
to the next gesture. **Q** quits.

Make each session look **different**, that's the whole point:

| Session | Where / how |
|---|---|
| 1 | Normal room light, plain wall behind you |
| 2 | Another spot: window or a different wall, lamp on/off, sleeves rolled up |
| 3 | Different again (evening light, standing further back, ask a friend to do it) |

For **none**: empty box, your arm passing through, a relaxed hand that's not a gesture, your face.

Wait at least a minute between sessions (the session name is the time).

## 2. Train both models (about 15-30 minutes on the laptop)

```powershell
..\train-venv\Scripts\python.exe train.py --model both --deploy best
```

This trains **our own CNN from scratch** and **MobileNetV2 transfer learning**
on the same photos, tests both on the newest session (photos they have never
seen), and copies the better one into `models\` for the Pi.

Results to look at:
- `models\comparison.png` and `models\comparison.txt`: the two models side by side
- `models\cnn\` and `models\mobilenet\`: training curves, confusion matrix, report

## 3. Send to the Pi and test live

```powershell
cd ..
powershell -ExecutionPolicy Bypass -File .\send-code-to-pi.ps1
powershell -ExecutionPolicy Bypass -File .\send-model-to-pi.ps1
powershell -ExecutionPolicy Bypass -File .\run-with-laptop-webcam.ps1 -Hardware
```
