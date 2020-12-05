import tensorflow as tf


class Reinforce:
    def __init__(self, environment, summary_writer=None):
        self.environment = environment
        self.observation_shape = self.environment.get_observation_shape()
        self.num_actions = self.environment.get_num_actions()
        self.step_size = 0.001
        self.discount_factor = 1
        self.policy = self.get_nn_policy()
        self.summary_writer = summary_writer

    def get_nn_policy(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=64, input_shape=self.observation_shape, activation='relu'))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.num_actions, activation='softmax'))

        return model

    def get_gradients(self, state, action):
        with tf.GradientTape() as tape:
            selected_action_log_prob = tf.math.log(self.policy(state)[0][action])

        eligibility_vector = tape.gradient(selected_action_log_prob, self.policy.trainable_variables)

        return eligibility_vector

    def update_gradients(self, gradients, episode_return, num_step):
        for i in range(len(self.policy.trainable_variables)):
            self.policy.trainable_variables[i].assign_add(
                self.step_size * episode_return * (self.discount_factor ** num_step) * gradients[i])

    def get_action(self, observation):
        action_probs = self.policy(observation)[0]
        action = tf.argmax(action_probs).numpy()

        return action

    def learn_optimal_policy(self, num_epochs=10000):
        for epoch_num in range(num_epochs):
            states = []
            actions = []
            rewards = []
            gradients_list = []

            done = False
            observation = self.environment.reset()

            while not done:
                observation = tf.expand_dims(observation, 0)
                states.append(observation)

                action = self.get_action(observation)
                actions.append(action)

                observation, reward, done, info = self.environment.step(action)
                rewards.append(reward)

                gradients = self.get_gradients(state=states[-1], action=actions[-1])
                gradients_list.append(gradients)

            if self.summary_writer:
                self.summary_writer.write_summary("Episode Return", sum(rewards), epoch_num)

            returns = rewards.copy()
            for i in reversed(range(len(rewards) - 1)):
                returns[i] += self.discount_factor * returns[i + 1]

            for i in range(len(states)):
                self.update_gradients(gradients_list[i], returns[i], i)
