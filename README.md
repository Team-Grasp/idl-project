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
 
