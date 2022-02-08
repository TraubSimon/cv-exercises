# Frequently Asked Questions.


## Setup: 

### Regarding bashprofile and bashrc:
- Problem: conda: Befehl nicht gefunden / Command not found
- Actual problem: .bashrc is not sourced when logging in via ssh
- see [bashrc-at-ssh-login](https://stackoverflow.com/questions/820517/bashrc-at-ssh-login)
- Solution:
- if .bash_profile does not exist next to your bash file then create it and paste the following
```
if [ -f ~/.bashrc ]; then
  . ~/.bashrc
fi
```

### Access data options
- work on pool pcs
- copy to local via vscode (right click -> download)
- scp
- mout the shared-data1 directory on your local machine using sshfs

### Access denied / Password issues
- Problem: login at [NextCloud](https://nc.informatik.uni-freiburg.de/index.php/apps/rainloop/) works but not when using ssh
- Solution: do not use Umlauts (äöüß etc) in you password. Different encodings in the browser (setting the password) and terminal lead to different hashes

### Home folder exceeds 1GB
- can lead to weird behaviour
- you cannot create new files
- list folder and file sizes in home: `cd ~`, `du --max-depth=1 -B M`
- you should be able to delete .cache (`rm -r .cache`) without negative impact
- also check your trash and .local and .share and your mailfolder if you can reduce them but be careful!
- **vscode-server can be large**
  - here is a nice solution to move it to your project space
  - it involves creating a symlink in your home directory
  - [move-vscode-server](https://stackoverflow.com/questions/62613523/how-to-change-vscode-server-directory)
- **pytorch pretrained models**
  - the pretrained weights are stored under `~/.cache/torch...`
  - change this directory in all python scripts
  - [see stackoverflow](https://stackoverflow.com/questions/52628270/is-there-any-way-i-can-download-the-pre-trained-models-available-in-pytorch-to-a)

### use shared conda environment
- your conda installation takes up a lot of space. In our case >5GB, which might exceed your storage quota.
- We share our environment `cvenv` which is sufficient to run the example solutions
- **Steps to use shared conda environment**
- edit your `~/.condarc` and add the following lines:
```envs_dirs:
  - /project/cv-ws2122/shared-data1/miniconda3/envs
```
- activate our environment: `conda activate cvenv`
- check if the output of  `which python` equals `/project/cv-ws2122/shared-data1/miniconda3/envs/cvenv/bin/python`

### Recommended: vscode - IDE for working on pool machines
#### steps to install vs-code (on linux without sudo)
- download https://code.visualstudio.com/docs/?dv=linux64
- you can probably skip the next steps if you installed via the package manager
- copy to <path_to_bin_and_install>/install
- extract `tar -xf code-stable-x64-1636111355.tar.gz`
- make a bin directory next to install i.e. <path_to_bin_and_install>/bin: `mkdir bin`
- cd bin
- create softlink: `ln -s ../install/VSCode-linux-x64/bin/code vscode`
- add to your bash to call vscode from everywhere: `export PATH="<path_to_bin_and_install>/bin:$PATH"`

#### VSCode - remote
- follow the steps in https://code.visualstudio.com/docs/remote/ssh
- open it with vscode from any terminal
- hit crtl p and paste `ext install ms-vscode-remote.vscode-remote-extensionpack`
- click install
- press f1 and click on add host
- enter the ssh command that you used from your local machine before to connect to the pool
- e.g. replace <username> with your username: ssh <username>@login.informatik.uni-freiburg.de
- then enter password, press f1 again and connect to host, and voila, you're on the tf pool
- click on open folder and enter the path to your project
- `/project/cv-ws2122/<username>/cv-exercises/`
- (in order to have the cv-exercises folder you need to do a `git clone https://github.com/lmb-freiburg/cv-exercises.git`)

#### VSCode Shortcuts:
	- https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf
	- https://code.visualstudio.com/shortcuts/keyboard-shortcuts-linux.pdf
	- https://code.visualstudio.com/shortcuts/keyboard-shortcuts-macos.pdf
	- open terminal: ctrl j
	- open settings: ctrl ,
	- open command palette: ctrl shift p
	- set python interpreter of your conda environment: 
		ctrl shift p 
		python select interpreter
		choose path to your conda environment
		complaints about not knowing numpy will be gone

### Remote SSH in vscode directly to a pool machine
Instead of connecting vscode to the login node via ssh and then connect in the terminal to one of the pool machines, one can directly connect via ssh to a pool machine as follows:
- Crtl+shift+p: then Remote-SSH: Open SSH Configuration File and choose your loca config file.
- Usually you will see an entry for every server like:
```
Host login.informatik.uni-freiburg.de
  HostName login.informatik.uni-freiburg.de
  User username
```
Since the pool machines are only accessible after connecting to login, you can add one more entry for a pool machine as follows:
```
Host tfpool21
  HostName tfpool21
  ProxyJump login.informatik.uni-freiburg.de
  User username
```
Then, when you select tfpool21, it will first connect to login and then to tfpool21, hence will ask for the password twice.

This is required when running jupyter notebook that requires libraries that are not installed on the login node (exercise 07).

### Tensorboard visualization in vscode
To run tensorboard, you run the command from the terminal:

`tensorboard --logdir path`

This will show a link and probably a window will pop-up with 'open-in-local-browser'

if it does not pop up: hover over link in cmd and click on 'follow link using forwarded port'

## Additional Material
### Batch Norm
- nice explanation of intuition and parameters
- [batchnorm-towardsdatascience](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)

