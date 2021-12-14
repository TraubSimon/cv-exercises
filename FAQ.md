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


## Additional Material
### Batch Norm
- nice explanation of intuition and parameters
- [batchnorm-towardsdatascience](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)