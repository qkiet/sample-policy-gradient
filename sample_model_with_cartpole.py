import tensorflow as tf
import numpy as np
import gymnasium as gym
import tensorflow_probability as tfp
import sys

class PolicyGradientSimpleModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(30,activation='relu')
        self.d2 = tf.keras.layers.Dense(30,activation='relu')
        self.out = tf.keras.layers.Dense(2,activation='softmax')

    def call(self, input_data):
        x = tf.convert_to_tensor(input_data)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        return x

class PolicyGradientAgent():
    def __init__(self):
        self.model = PolicyGradientSimpleModel()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 1

    def act(self,state):
        prob = self.model(np.array([state]))
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def a_loss(self,prob, action, reward):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*reward
        return loss

    def train(self, states, rewards, actions):
        sum_reward = 0
        discnt_rewards = []
        rewards.reverse()
        for r in rewards:
            sum_reward = r + self.gamma*sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()

        for state, reward, action in zip(states, discnt_rewards, actions):
            with tf.GradientTape() as tape:
                p = self.model(np.array([state]), training=True)
                loss = self.a_loss(p, action, reward)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))





def train_agent(agent, num_of_episodes):
    def train_env_setup():
        env= gym.make("CartPole-v1")
        return env

    env = train_env_setup()
    for ep in range(num_of_episodes):
        done = False
        state = env.reset()[0]
        total_reward = 0
        rewards = []
        states = []
        actions = []
        while not done:
            #env.render()
            action = agent.act(state)
            #print(action)
            next_state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            total_reward += reward

            if done:
                agent.train(states, rewards, actions)
                #print("total step for this episord are {}".format(t))
                print("total reward after {} steps is {}".format(ep, total_reward))

def test_agent(agent, test_episodes=5):
    def test_env_setup():
        env= gym.make("CartPole-v1", render_mode="human")
        return env
    for ep in range(test_episodes):
        env = test_env_setup()
        state = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                print("done with total reward {}".format(total_reward))
                break

if __name__ == "__main__":
    agent = PolicyGradientAgent()
    train_agent(agent, int(sys.argv[1]))
    test_agent(agent, 10)
