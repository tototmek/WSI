import numpy as np
import gym

def q_learning(env : gym.Env, t_max, gamma, beta, epochs):
    actions_count = env.action_space.n
    states_count = env.observation_space.n

    # Inicjalizacja losową strategią
    Q = np.random.random((states_count, actions_count))
    epoch = 0
    reward = 0

    while epoch < epochs:
        t = 0
        state = env.reset()
        is_terminal = False
        while t < t_max and not is_terminal:
            # Wybór akcji na podstawie aktualnej strategii
            action = choose_action(env, Q, state)
            # Wykonanie akcji
            new_state, reward, is_terminal, _, _ = env.step(action)

            max_score = -np.inf
            for a in range(actions_count):
                score = Q[new_state, a]
                max_score = max(score, max_score)

            # Poprawienie strategii
            delta = reward + gamma * max_score - Q[state, action]

            # To do mocnej poprawki. Nie działa na tym etapie
            Q[state, action] = Q[state, action] + beta * delta

            state = new_state
            t += 1
        epoch += 1


def choose_action(env : gym.Env, Q, state):
    e = 0.1
    if np.random.random() < e:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state, :])

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="ansi")
    #q_learning(env, 15, 0.9, 0.1, 0.5)

