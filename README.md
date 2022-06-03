# DQNScoresFunction

## Installation

Run that command in order to install python dependencies (use `workon` command to enable virtual environment first):

```bash
make install-deps
```

## Usage

Run from the root directory these commands:

```bash
# run on samples from the `data` directory
make run-image 
```

```bash
# run with the stream from camera (camera index is in the src/config/config.py)
# press `q` key to stop the inference
make run-video 
```