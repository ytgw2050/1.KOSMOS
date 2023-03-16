import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class CustomAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  # discount rate
        self.learning_rate = 0.01
        self.states = []
        self.actions = []
        self.rewards = []
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))  
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        action_values = self.model.predict(state,verbose = 0)
        return action_values.flatten()

    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def train(self):
        episode_length = len(self.states)
        discounted_rewards = self._discount_and_normalize_rewards()

        X = np.vstack(self.states)
        Y = np.array(self.actions) * discounted_rewards.reshape(-1, 1)  # reshape 보상 to (N, 1)



        self.model.train_on_batch(X, Y)
        self.states, self.actions, self.rewards = [], [], []

    def _discount_and_normalize_rewards(self):
        discounted_rewards = np.zeros_like(self.rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_rewards[t] = running_add

        # Normalize
        if np.std(discounted_rewards) != 0:
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards




import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

class ActorCriticAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.actions = []
        self.rewards = []

        self.actor, self.critic = self._build_actor_critic()
        self.actor_optimizer = Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = Adam(learning_rate=self.learning_rate)

    def _build_actor_critic(self):
        state_input = Input(shape=(self.state_size,))
        x = Dense(24, activation='relu')(state_input)
        x = Dense(24, activation='relu')(x)

        # Actor model
        probs = Dense(self.action_size, activation='softmax')(x)

        # Critic model
        values = Dense(1, activation='linear')(x)

        actor = Model(inputs=state_input, outputs=probs)
        critic = Model(inputs=state_input, outputs=values)

        return actor, critic

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        policy = self.actor.predict(state, verbose=0).flatten()
        return policy #np.random.choice(self.action_size, p=policy)

    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


    def train(self):
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.float32)  # 액션을 그대로 사용

        discounted_rewards = self._discount_and_normalize_rewards()

        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            values = self.critic(states)
            policy = self.actor(states)
            advantages = discounted_rewards - tf.squeeze(values)

            # 액션 자체를 사용하여 actor_loss를 계산하기
            actor_loss = -tf.reduce_mean(tf.reduce_sum(actions * tf.math.log(policy + 1e-10), axis=1) * advantages)
            critic_loss = tf.reduce_mean(tf.math.square(values - discounted_rewards))

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        self.states, self.actions, self.rewards = [], [], []

    def _actor_loss(self, actions, states, advantages):
        policy = self.actor(states)
        action_prob = tf.reduce_sum(actions * policy, axis=1)
        return -tf.reduce_mean(tf.math.log(action_prob + 1e-10) * advantages)

    def _discount_and_normalize_rewards(self):
        discounted_rewards = np.zeros_like(self.rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_rewards[t] = running_add

        mean = np.mean(discounted_rewards)
        std = np.std(discounted_rewards)
        discounted_rewards = (discounted_rewards - mean) / (std + 1e-7)

        return discounted_rewards
