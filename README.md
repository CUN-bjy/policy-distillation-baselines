# policy-distillation-baselines

Pytorch Implementation of Policy Distillation for control, which has well-trained teachers via [stable_baselines3](https://github.com/DLR-RM/stable-baselines3).



STATUS : [`IN PROGRESS`](https://github.com/CUN-bjy/policy-distillation-baselines/projects)



#### Notice

> *This repository is based on [Mee321/policy-distillation](https://github.com/Mee321/policy-distillation) and integrated with [DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) environment.*



![](./docs/pd_baselines_figures-Page-2.svg)

[**`ZOOM In`**](https://raw.githubusercontent.com/CUN-bjy/policy-distillation-baselines/main/docs/pd_baselines_figures-Page-2.svg)



## Installation

```bash
git clone https://github.com/CUN-bjy/policy-distillation-baselines.git
cd policy-distillation-baselines
virtualenv venv
source venv/bin/active
venv/bin/pip install -r requirements.txt
```

You don't need to use virtual environment but recommended.

With every moment of using this package, you should source the `venv`. plz  `source venv/bin/active`.



## Enjoy a Trained Agent

```bash
python classroom.py --algo algo_name --env env_id
# example) python classroom.py --algo td3 --env AntBulletEnv-v0
```

You can just play by `python classroom.py`, default by model is `td3`, env is `AntBulletEnv-v0 `.

See the details on this [link](https://github.com/DLR-RM/rl-baselines3-zoo#enjoy-a-trained-agent).



## Policy Distillation

Distillation from trained teacher agent to pure student agent.

*(I only tested on TD3, AntBulletEnv-v0 environment  so that I cannot not sure running other algorithms.* 

**PR is wellcome**!)

```bash
python policy_distillation.py
```





## References

[1] 

```
@misc{rusu2016policy,
      title={Policy Distillation}, 
      author={Andrei A. Rusu and Sergio Gomez Colmenarejo and Caglar Gulcehre and Guillaume Desjardins and James Kirkpatrick and Razvan Pascanu and Volodymyr Mnih and Koray Kavukcuoglu and Raia Hadsell},
      year={2016},
      eprint={1511.06295},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

[2] [Mee321/policy-distillation](https://github.com/Mee321/policy-distillation)

[3] [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) / [DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) / [DLR-RM/rl-trained-agents](https://github.com/DLR-RM/rl-trained-agents)