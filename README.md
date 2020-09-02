**Status:** Archive (code is provided as-is, no updates expected)

# Learning to Summarize from Human Feedback

This repository contains code to run our models, including the supervised baseline, the trained reward model, and the RL fine-tuned policy.

Supported platform: Python 3.7 64-bit on Ubuntu 18.04

## Install

- Install [pipenv](https://github.com/pypa/pipenv#installation).

- Clone this repo.  Then, inside it:
  ```
  pipenv install
  ```

## Run the models

You'll need to run this on a machine with an Nvidia GPU.

First, let's run some tests to make sure everything is working.
```
pipenv run exps/sample.py test test-sample
pipenv run exps/eval_rm.py test test-eval
```

Now let's run some actual evaluations. We can have the model summarize some posts from the validation set:
```
pipenv run exps/sample.py ppo_xl sample-ppo-xl --num_queries 32
```
This will output to `/tmp/jobs/sample-ppo-xl/results/`.

Now we can evaluate them using the reward model:
```
pipenv run exps/eval_rm.py rm4 eval-rm4 --input_path /tmp/jobs/sample-ppo-xl/results/ 
```
This will print some aggregate statistics and output scores for each sample to `/tmp/jobs/eval-rm4/results/`.

