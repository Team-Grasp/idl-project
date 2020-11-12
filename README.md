# Requirements
1. Ubuntu 18.04 or above
2. python 3.6+

# Setup 
We recommand using Conda env for the following installation. 

Download and extract Copeliasim from [here](https://www.coppeliarobotics.com/downloads).

Next, follow installation instructions these repositories,
1. [PyRep](https://github.com/Team-Grasp/PyRep)
2. [RLBench](https://github.com/Team-Grasp/RLBench)
3. [Stable Baselines 3](https://github.com/Team-Grasp/stable-baselines3)


Next, clone this repository. 
```sh
$ git clone https://github.com/Team-Grasp/idl-project.git
```

# Common installation issues
1. Problem: 
    ```sh
    Traceback (most recent call last):
    File "main.py", line 108, in <module>
        model.learn(total_timesteps=total_timesteps, callback=callback) 
    File "/home/deval/anaconda3/envs/drl/lib/python3.7/site-packages/stable_baselines3/ppo/ppo.py", line 265, in learn
        reset_num_timesteps=reset_num_timesteps,
    File "/home/deval/anaconda3/envs/drl/lib/python3.7/site-packages/stable_baselines3/common/on_policy_algorithm.py", line 240, in learn
        logger.dump(step=self.num_timesteps)
    File "/home/deval/anaconda3/envs/drl/lib/python3.7/site-packages/stable_baselines3/common/logger.py", line 379, in dump
        Logger.CURRENT.dump(step)
    File "/home/deval/anaconda3/envs/drl/lib/python3.7/site-packages/stable_baselines3/common/logger.py", line 544, in dump
        _format.write(self.name_to_value, self.name_to_excluded, step)
    File "/home/deval/anaconda3/envs/drl/lib/python3.7/site-packages/stable_baselines3/common/logger.py", line 143, in write
        self.file.write("\n".join(lines) + "\n")
    ValueError: I/O operation on closed file.
    ```
    Solution: Add the following line to the logger.py file in the installation. Alternatively you can change the line and install the library locally. 

    Add the following line before line [142](https://github.com/DLR-RM/stable-baselines3/blob/e2b6f5460f362ecad3777d6fe2950f3199058d8f/stable_baselines3/common/logger.py#L142).
    ```py
    if not self.own_file: self.file = sys.stdout
    ```

2. Problem: 
    ```
    qt.qpa.plugin: Could not find the Qt platform plugin "xcb" in "<python install dir>/lib/python3.6/site-packages/cv2/qt/plugins"
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

    Aborted (core dumped)
    ```
    Solution: Replace existing libqxc.so file from the Copeliasim directory.
    ```
    sudo cp <copeliasim install dir>/libqxcb.so <python install dir>/lib/python3.6/site-packages/cv2/qt/plugins/platforms/
    ```

# Run Random Episode 

```sh
$ cd idl-project
$ python main.py --render --eval
```

# Train model
```sh
$ cd idl-project
$ python main.py --train
```


# Run Trained model 
```sh
$ cd idl-project
$ python main.py --render --eval --model_path <path to saved model>
```
