import gym 
from gym.envs.registration import register
import platform

def get_key_in_str():
    OS = platform.system()
    if OS == "Linux":
        from getch import getch     # Ubuntu
        return getch()
        
    elif OS == "Windows":
        from msvcrt import getch    # Windows
        return getch().decode()

# key mapping
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    'w' : UP,
    's' : DOWN,
    'd' : RIGHT,
    'a' : LEFT
}

register(
    id = 'FrozenLake-v3', 
    entry_point='gym.envs.toy_text:FrozenLakeEnv', 
    kwargs={
        'map_name' : '4x4',
        'is_slippery' : False
    }
)

env = gym.make("FrozenLake-v3", render_mode="rgb_array")
env.reset()
env.render()

while True:
    key = get_key_in_str()
    if key not in arrow_keys.keys():
        print("Wrong key... Game aborted!")
        break
    action = arrow_keys[key]
    state, reward, done, _, info = env.step(action)
    env.render()
    print(f"State : {state}, Action : {action}, Reward : {reward}, Info : {info}")

    if done :
        print(f"Finished with reward {reward}")
        break