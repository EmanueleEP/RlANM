from gymnasium.envs.registration import register


# register(
#     id='ANM6_easy_test',
#     entry_point='gym_anm.envs:ANM6_easy_test',
# )

register(
    id='Ciao-v0',
    entry_point='env.ciao.ciao_test:Ciao_test',
)