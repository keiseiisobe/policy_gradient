import numpy as np
import numpy.typing as npt
import gymnasium as gym
import ale_py
import _pickle as pickle
import os

gym.register_envs(ale_py)
x_dim = 80 * 80
h_dim = 200 * 1


class Model:
    def __init__(self, X, H):
        if os.path.isfile("model.p"):
            self.model = pickle.load(open('model.p', 'rb'))
            print("model.p loaded")
        else:
            self.model = {}
            self.model["W1"] = np.random.randn(H, X) / np.sqrt(X)
            self.model["W2"] = np.random.randn(H) / np.sqrt(H)
        self.grad_buffer = { k : np.zeros_like(v) for k, v in self.model.items() }
        #self.rmsprop_cache = { k : np.zeros_like(v) for k, v in self.model.items() }

    def sigmoid(self, var):
        return 1.0 / (1.0 + np.exp(-var))

    def prepro(self, obs):
        obs = obs[35:195]
        obs = obs[::2,::2,0]
        obs[obs == 144] = 0
        obs[obs == 109] = 0
        obs[obs != 0] = 1
        return obs.astype(np.float64).ravel()
    
    def policy_forward(self, x: npt.NDArray[np.float64]) -> tuple[np.float64, npt.NDArray[np.float64]]:
        h = np.dot(self.model["W1"], x)
        h[h < 0] = 0
        y = np.dot(self.model["W2"], h)
        prob = self.sigmoid(y)
        return prob, h

    def discount_rewards(self, eprs):
        running_rewards = 0
        discounted_rewards = np.zeros_like(eprs)
        for i in reversed(range(0, eprs.size)):
            if eprs[i] != 0:
                running_rewards = 0
            discounted_rewards[i] = running_rewards + eprs[i]
            running_rewards = discounted_rewards[i] * gamma
        return discounted_rewards

    def backprop(self, epdlogps, eprs, ephs, epxs):
        # Loss = logp(action="UP" | x)
        # d = dL / dz(2) = 1 - sigmoid(z(2))
        epdrs = self.discount_rewards(eprs)
        epdrs -= np.mean(epdrs)
        epdrs /= np.std(epdrs)
        epdlogps *= epdrs
        dW2 = np.dot(ephs.T, epdlogps).ravel()
        dh = np.outer(epdlogps, self.model["W2"])
        dh[ephs <= 0] = 0
        dW1 = np.dot(dh.T, epxs)
        return {"W1": dW1, "W2": dW2}


episodes = 0
total_rewards = 0
gamma = 0.99
learning_rate = 0.0001
batch_size = 10
prev_x = None
decay_rate = 0.99
xs, hs, rs, dlogps = [], [], [], []

if __name__ == "__main__":
    mymodel = Model(x_dim, h_dim)
    env = gym.make("ALE/Pong-v5")#, render_mode="human")
    obs, info = env.reset(seed=42)
    while True:
        cur_x = mymodel.prepro(obs)
        x = cur_x - prev_x if prev_x is not None else np.zeros(x_dim)
        prev_x = cur_x
        prob, h = mymodel.policy_forward(x)
        action = 2 if np.random.uniform() < prob else 3
        obs, reward, terminated, truncated, info = env.step(action)
        xs.append(x)
        hs.append(h)
        rs.append(reward)
        y = 1 if action == 2 else 0
        dlogps.append(y - prob)
        total_rewards += reward
        if terminated or truncated:
            episodes += 1
            # (N,) -> (1, N) for each x, h, reward and prob
            epxs = np.vstack(xs)
            ephs = np.vstack(hs)
            eprs = np.vstack(rs)
            epdlogps = np.vstack(dlogps)
            xs, hs, rs, dlogps = [], [], [], []
            
            dW = mymodel.backprop(epdlogps, eprs, ephs, epxs)

            for k in mymodel.model:
                mymodel.grad_buffer[k] += dW[k]
            if episodes % batch_size == 0:
                for k in mymodel.model:
                    g = mymodel.grad_buffer[k]
                    #mymodel.rmsprop_cache[k] = decay_rate * mymodel.rmsprop_cache[k] + (1 - decay_rate) * g**2
                    mymodel.model[k] += learning_rate * g# / (np.sqrt(mymodel.rmsprop_cache[k]) + 1e-5)
                    mymodel.grad_buffer[k] = np.zeros_like(g)
            if episodes % 10 == 0:
                pickle.dump(mymodel.model, open("model.p", "wb"))
                print("episode:", episodes, "total reward: ", total_rewards)
                total_rewards = 0
            obs, info = env.reset()
            prev_x = None
    env.close()
