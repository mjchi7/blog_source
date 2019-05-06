# Running Dockerized Wine with Mac OS X

Dockerized application is awesome no matter which role you are in: developer, sysadmin, devOps, and etc. For me as a developer, with so many neatly packaged, dockerized applications means that I can write and develop in my favourite **Mac OS X** environment, while being able to build and execute (test) my application for **Windows** user. 

One of my latest challenge is when I have to execute a GUI based Windows application (.exe). That's when I started to look at a possible solution, with which I can run a GUI based `.exe` program through docker. That's when I stumbled upon the neatly [dockerized `Wine` application]([https://hub.docker.com/r/scottyhardy/docker-wine/](https://hub.docker.com/r/scottyhardy/docker-wine/)), which is an application frequent by Linux user to run Windows program. Running a console program itself is not an issue and can be done by following the instruction given. Pretty straightforward! 

However, things get a little bit trickier when you need to run a GUI application because you will need to hook up a working X11 server in order to display the windows properly. ScottyHardy (the author) has actually provided instruction on how to hook up a X11 server connection to the container, *however, it only works for Linux (for detailed explanation refers to Appendix)*. 

If you are running the said container with Mac OS X, with the instruction given, you will get the following error message:

```shell
1:~$ wine notepad.exe
0009:err:winediag:nodrv_CreateWindow Application tried to create a window, but no driver could be loaded.
0009:err:winediag:nodrv_CreateWindow Make sure that your X server is running and that $DISPLAY is set correctly.
```
 
 The cause of this error is because `Wine` couldn't find any display driver. Effort in debugging the variable `$DISPLAY` will gone to waste because if you have installed your XQuartz properly (through HomeBrew or `.dmg`), you can see that doing `echo $DISPLAY` will give you the display socket. 

To circumvent this obstacles, I am lucky enough to come across this life-saver [blogpost.](https://sourabhbajaj.com/blog/2017/02/07/gui-applications-docker-mac/)

The following steps to rectify the issues are all taken from the blog link above and all credits were to go to the owner.

There are 3 additional steps need to be taken as compared to if you are running Docker from Linux distro.

1. Allow external connection in XQuartz
	- Open up XQuartz and head to Preference -> Security
	- Tick the option "Allow connections from network clients"

2. Adding host machine IP to `xhost`
	- Get the ip of the machine and store it in a variable `IP` by executing the following command
	`IP=$(ipconfig en0 | grep inet | awk '$1=="inet" {print $2}')`
	- Add the host machine `IP` to the list of trusted client machine by executing the following command
	`xhost + $IP`
	- You should see the terminal output the message `192.168.xxx.xxx being added to access control list`

3. Setting the `$DISPLAY` variable correctly
	- Contrary to the instruction given in the docker image README, instead of setting the environment variable `DISPLAY` as it is `--env="DISPLAY"`, do the following instead
	`--env DISPLAY=$IP:0`

Voila! you got yourself a working dockerized Wine application in Mac OS X environment.

## Summary
- For the dockerized wine application, head to [ScottyHardy/docker-wine](https://hub.docker.com/r/scottyhardy/docker-wine/)
- For the steps needed in order to make dockerized wine GUI works, head to [ Running GUI applications using Docker for Mac](https://sourabhbajaj.com/blog/2017/02/07/gui-applications-docker-mac/)
- The main thing is to set the `$DISPLAY` environment variable by including the host machine IP.

## Appendix
### Why Mac OS X need all these additional steps.

The main problem lies in the fact that Docker is not running as a native process in Mac OS X. Instead, it is running on top of a VM call [Hyperkit](https://docs.docker.com/docker-for-mac/docker-toolbox/). With this additional layer of VM, the flag `--network="host"` will expose the containers to the network infrastructure of that VM, and not the Mac OS X's. 

For more in depth discussion regarding this issue, refers to the following [forum post.](https://forums.docker.com/t/should-docker-run-net-host-work/14215/21)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI0NTM5NjMyOF19
-->