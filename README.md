### Installing Dependencies

### Running Training
Before running any training, create the following folders:
```
mkdir models && mkdir runs
```
Next, run training. Other hyperparameters such as max episode length and number of episodes to generate for one iteration can be set in main.py
Early-termination and special penalties for invalid actions can also be specified in main.py.
```
python3 main.py --train --lr=3e-4
```
The above command will produce two folders: 
```
models/<some_timestamp>
runs/PPO_<id>
```
The first folder will have saved weights that can be loaded during evaluation. 
The second folder will contain the tensorboard metrics that can visualized with the following command:
```
tensorboard --logdir runs/PPO_<id>
```

To run evluation and visualize a certain step of training:
```
python3 main.py --eval --render --model_path=models/<some_timestamp>/<some_train_step>.zip
```
### Results
You can see some results of trained agents using different parameters and reward-shaping techniques:
https://drive.google.com/file/d/1DyxaBclH8GbvDGLKfUSSwWXZ2QThympe/view?usp=sharing


### Issues
1. When running an RLBench simulator after importing stablebaslines3, an error popped up: 
```
qt.qpa.plugin: Could not find the Qt platform plugin "xcb" in "/home/alvin/.local/lib/python3.6/site-packages/cv2/qt/plugins"
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Aborted (core dumped)
```

Solution: Replace existing libqxc using the following command:
```
sudo mv /usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqxcb.so /home/alvin/.local/lib/python3.6/site-packages/cv2/qt/plugins/platforms/
```
 
