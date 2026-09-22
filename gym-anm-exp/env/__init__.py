from gymnasium.envs.registration import register

register(
    id='Ciao-v0',
    entry_point='env.ciao.ciao_test:Ciao_test',
)

register(
    id='BCP-v0',
    entry_point='env.BCP.bcp_example:BCP',
)


register(
    id='Custom_ANM6Easy-v0',
    entry_point='env.custom_anm6_env.anm6_easy_custom:ANM6Easy_Custom',
)

register(
    id='RealProfileANM6-v0',
    entry_point='env.custom_anm6_env.real_profiles:RealProfileANM6',
)
