phaser: The weapon of choice for ptychographic reconstructions
---

To install, first clone folder from github 

**In terminal:**

```
conda create -n phaser python=3.12
```

```
source activate phaser
```

```
conda install pip
conda install matplotlib
```

```
cd to phaser (cloned folder from github)
```

```
pip install -e '.[jax, web]'
```

**If npm not already installed, run the following or otherwise skip**
```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
```

```
cd phaser/web  
npm install 
npm run build
```

```
cd path/to/yaml_reconstruction_file
```

```
phaser serve 
```

**Launch Chrome**

http://localhost:5050

start local worker
submit name of reconstruction file: e.g. test_reconstuction.yaml

