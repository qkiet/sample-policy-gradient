from pygame import init
import tensorflow as tf
import gymnasium as gym

class PolicyGradientSimpleModel(tf.keras.Model):
    def __init__(self, action_size=2):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(30,activation='relu')
        self.d2 = tf.keras.layers.Dense(30,activation='relu')
        self.out = tf.keras.layers.Dense(action_size)

    def call(
            self,
            input_data: tf.Tensor,
            ) -> tf.Tensor: 
        x = self.d1(input_data)
        x = self.d2(x)
        x = self.out(x)
        return x

env= gym.make("CartPole-v1")

def env_step(action) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Returns state, reward and done flag given an action."""
    state, reward, done, _, _ = env.step(action)
    return (tf.convert_to_tensor(state, dtype=tf.float32),
            tf.convert_to_tensor(int(reward), dtype=tf.int32),
            tf.convert_to_tensor(done, dtype=tf.int32))

def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps_of_episode: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps_of_episode):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities
        action_logits_t = model(state)
        
        # Sample next action from the action probability distribution
        action = tf.random.categorical(action_logits_t, 1).numpy()[0,0]
        action_probs_t = tf.nn.softmax(action_logits_t)
        print("action_logits_t: ", action_logits_t)
        print("actions_probs_t: ", action_probs_t)
        print("chosen action {} with prob {}".format(action, action_probs_t[0, action]))

        # Store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        state, reward, done = env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    rewards = rewards.stack()
    print("action_probs: ", action_probs)
    print("rewards: ", rewards)
    return action_probs, rewards

if __name__ == "__main__":
    initial_state = env.reset()[0]
    initial_state_tf = tf.constant(initial_state, dtype=tf.float32)
    model = PolicyGradientSimpleModel(action_size=env.action_space.n)
    max_steps_of_episode = 1000
    run_episode(initial_state_tf, model, max_steps_of_episode)


# @tf.function
# def train_for_episode(
#         initial_state: tf.Tensor,
#         model: tf.keras.Model,
#         optimizer: tf.keras.optimizers.Optimizer,
#         gamma: float,
#         max_steps_of_episode: int,
#         ) -> tf.Tensor:
#     """Runs a model training step."""

#     with tf.GradientTape() as tape:

#         # Run the model for one episode to collect training data
#         action_probs, values, rewards = run_episode(
#             initial_state, model, max_steps_per_episode)

#         # Calculate the expected returns
#         returns = get_expected_return(rewards, gamma)

#         # Convert training data to appropriate TF tensor shapes
#         action_probs, values, returns = [
#             tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

#         # Calculate the loss values to update our network
#         loss = compute_loss(action_probs, values, returns)

#     # Compute the gradients from the loss
#     grads = tape.gradient(loss, model.trainable_variables)

#     # Apply the gradients to the model's parameters
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))

#     episode_reward = tf.math.reduce_sum(rewards)

#     return episode_reward

