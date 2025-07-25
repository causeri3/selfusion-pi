# Selfusion
### Repository for an art installation  
*(made for Fusion / Bornhack 2025)*

---
*Bornhackers*: Check out http://151.216.66.159:8000/docs

---

Takes automated selfies, generates more pictures from that, and plays a bouncing gif. 

---
#### Online Mode Example
https://github.com/user-attachments/assets/008ab640-0a8f-4742-81ea-54a7783c44f8

#### Offline Mode Examples
<img src="https://github.com/user-attachments/assets/a17bf85d-27f7-4f19-9231-2703db1c9837" width="300"/>
<img src="https://github.com/user-attachments/assets/29be4ae7-0691-4209-a8de-552207df9c77" width="300"/>
<img src="https://github.com/user-attachments/assets/6d96d362-f0e3-45b4-bb49-d5ce5d6ce572" width="300"/>


#### Waiting Screens
<img width="300" alt="Screenshot 2025-07-03 at 08 17 57" src="https://github.com/user-attachments/assets/e3fb49f1-9c1f-4b36-a833-9721a986f01b" />

<img width="300" alt="Screenshot 2025-07-03 at 08 43 01" src="https://github.com/user-attachments/assets/09dd3baf-1b9b-4031-8606-a2e41b961b8f" />

---

Using:
- **Automated Selfies**: Yolov8 trained by [arnabdhar](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)
- **Offline image generation**: Neural Style Transfer [deepeshdm](https://github.com/deepeshdm/PixelMix/tree/main)
- **Online image generation**: [Stable Diffusion XL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) in repo [sdxlturbo-api](https://github.com/causeri3/sdxlturbo-api)

---
## Settings / Args
See all arguments:
* with uv `uv run selfusion.py --help`
* standard `python selfusion.py --help`

Most interesting settings you will find in `selfusion_utils.args`, such as:  
- gif delay  
- waiting time  
- loading bar time  
- come closer screen or not
- face size threshold
- yolo settings (confidence, IoU)
- diffuser settings (strength, inference steps, prompt ...)


---
Another bit that's might be fun: Adding more style pictures — they are picked randomly.  
You can find them in `neural_style_transfer.style_images`.

---
## Hardware
- Raspberry Pi 5 (16GB)

---
## Software
### Dependencies

#### SDXLTurbo API
It's meant to work with [sdxlturbo-api](https://github.com/causeri3/sdxlturbo-api) hosted on some machine (runs well with mps on my Mac or cuda on a cloud VM with GPU - if you want you can test just the sdxlturbo code straight on Colab with T4, was super easy).

Anyway, it still has offline functionality. If it cannot get results from the Stable diffusion API, it generates pictures with neural style transfer on the local machine / the pi.


#### Linux

```bash
sudo apt-get install libgtk2.0-dev pkg-config
```

#### Python
The raspberry had `python 3.11` preinstalled

```bash
pip install -r requirements.txt
```
Gets it running.

Or even better if you use [uv](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1), run in the directory of this repo:
```sh
uv venv --python 3.11
uv pip install -r requirements.txt
```
**Note**:
Some of the code in this repo was used for experimentation (such as the `stable-diffusion` folder or `yolo_onnx_openvino.py`, they need more dependencies, so most directories have their own requirements files)

---
## Run
* with uv
 `uv run selfusion.py`

* standard
 `python selfusion.py`

### Systemd

I run it as systemd service:
```
[Unit]
Description=Fusion Raspberry Process
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=/home/pi/src/raspberry-pi
ExecStart=/home/pi/src/selfusion_venv/bin/python /home/pi/src/selfusion/selfusion.py
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/pi/.Xauthority

[Install]
WantedBy=default.target
```
that file goes under:
`~/.config/systemd/user/fusion.service`

**run service**
```bash
systemctl --user daemon-reload
systemctl --user enable fusion.service
systemctl --user start fusion.service
```



---
This project is licensed under AGPL v3+
