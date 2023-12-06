import tensorflow as tf
import numpy as np
import tf_util as U

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


class History:
    def __init__(self, history):
        self.history = history


def unison_shuffled(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


class AzeroBrain:
    def __init__(self, input_dim, output_dim, network_type="FC", lr=0.005, scope_name="", pv_loss_ratio=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.pv_loss_ratio = pv_loss_ratio

        if scope_name == "":
            # np.random.seed()
            seed = np.random.randint(10000)
            scope_name = "worker_" + str(seed)
        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name):
            self.X = tf.keras.Input(shape=input_dim, dtype=tf.float32, name='inpu')
            if network_type == "FC":
                self.model = self.compile_fully()
            if network_type == "QC":
                self.model = self.compile_fully_qc()
            if network_type == "QC2":
                self.model = self.compile_fully_qc2()
            if network_type == "CNN":
                assert len(input_dim) == 3, "Need 3 dimensional states to apply convolutions"
                self.model = self.compile_cnn_2()
            self.policy_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.output_dim))
            self.value_ph = tf.placeholder(dtype=tf.float32, shape=(None, 1))
            self.value_loss = (self.value_ph - self.value) ** 2
            self.policy_loss = tf.reduce_mean(-self.policy_ph * tf.log(self.policy + 1e-07), axis=-1)
            loss = tf.reduce_mean(self.value_loss + self.pv_loss_ratio * self.policy_loss)
            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.optimize_expression = self.optimizer.minimize(loss, var_list=self.vars)
            self._predict = U.function(inputs=[self.X],
                                       outputs=[self.policy, self.value])
            self._fit = U.function(inputs=[self.X, self.policy_ph, self.value_ph],
                                   outputs=[loss, tf.reduce_mean(self.policy_loss), tf.reduce_mean(self.value_loss)],
                                   updates=[self.optimize_expression])
            self._evaluate = U.function(inputs=[self.X, self.policy_ph, self.value_ph],
                                        outputs=[loss, tf.reduce_mean(self.policy_loss),
                                                 tf.reduce_mean(self.value_loss)])
            self.set_from_flat = U.SetFromFlat(self.vars)
            U.initialize()
        self.model = self

    def compile_fully_qc(self):
        x = self.X
        x = tf.layers.Dense(16, kernel_initializer='lecun_normal', activation='selu')(x)
        x1 = tf.layers.Dense(8, kernel_initializer='lecun_normal', activation='selu')(x)
        x2 = tf.layers.Dense(4,  kernel_initializer='lecun_normal', activation='selu')(x)

        out_1 = self.policy = tf.layers.Dense(self.output_dim, activation="softmax", kernel_initializer='glorot_normal',
                                              name="probabilities")(x1)
        out_2 = self.value = tf.layers.Dense(1, activation="linear", kernel_initializer='glorot_normal',
                                             name="value")(x2)
        return out_1, out_2

    def compile_fully_qc2(self):
        x = self.X
        x = tf.layers.Dense(64, kernel_initializer='lecun_normal', activation='selu')(x)
        x = tf.layers.Dense(32, kernel_initializer='lecun_normal', activation='selu')(x)
        x = tf.layers.Dense(16, kernel_initializer='lecun_normal', activation='selu')(x)
        x1 = tf.layers.Dense(16, kernel_initializer='lecun_normal', activation='selu')(x)
        x2 = tf.layers.Dense(8, kernel_initializer='lecun_normal', activation='selu')(x)

        out_1 = self.policy = tf.layers.Dense(self.output_dim, activation="softmax", kernel_initializer='glorot_normal',
                                              name="probabilities")(x1)
        out_2 = self.value = tf.layers.Dense(1, activation="linear", kernel_initializer='glorot_normal',
                                             name="value")(x2)
        return out_1, out_2

    def compile_fully(self):

        x = self.X
        x = tf.layers.Dense(20, kernel_initializer='lecun_normal', activation='selu')(x)
        x1 = tf.layers.Dense(10, kernel_initializer='lecun_normal', activation='selu')(x)
        x2 = tf.layers.Dense(4,  kernel_initializer='lecun_normal', activation='selu')(x)
        out_1 = self.policy = tf.layers.Dense(self.output_dim, activation="softmax", kernel_initializer='glorot_normal',
                                              name="probabilities")(x1)
        out_2 = self.value = tf.layers.Dense(1, activation="linear", kernel_initializer='glorot_normal',
                                             name="value")(x2)
        return out_1, out_2

    def compile_cnn_2(self):

        x = self.X
        strides = [1, 1, 2] #, 1, 1, 2
        units = [64, 64, 64]
        # strides = [1, 1, 2, 1, 1, 2]
        # units = [32, 64, 64, 64, 64, 64]
        for i in range(len(strides)):
            x = tf.layers.Conv2D(units[i], kernel_size=3, strides=strides[i], padding='same')(x)
            x = tf.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)

        x = tf.layers.Flatten()(x)
        x = tf.layers.Dense(128, kernel_initializer='lecun_normal', activation='relu')(x)
        x1 = tf.layers.Dense(128, kernel_initializer='lecun_normal', activation='relu')(x)
        x1 = tf.layers.Dense(64, kernel_initializer='lecun_normal', activation='relu')(x1)
        x2 = tf.layers.Dense(128, kernel_initializer='lecun_normal', activation='relu')(x)
        x2 = tf.layers.Dense(64, kernel_initializer='lecun_normal', activation='relu')(x2)

        out_1 = self.policy = tf.layers.Dense(self.output_dim, activation="softmax", kernel_initializer='glorot_normal',
                                      name="probabilities")(x1)
        out_2 = self.value = tf.layers.Dense(1, kernel_initializer='glorot_normal', name="value")(x2)
        return out_1, out_2
    
    def train(self, data, epochs, batch_size, stopping=False, verbose=0):
        x = np.asarray([s[0] for s in data]) # state
        y1 = np.asarray([s[1] for s in data]) # probability
        y2 = np.asarray([s[2] for s in data]) # value
        
        return self.fit(x, y1, y2, epochs, batch_size, stopping, verbose=verbose)

    def fit(self, states, policy, value, epochs, batch_size, stopping, verbose=0):
        states = np.reshape(states, (-1,) + self.input_dim)
        policy = np.reshape(policy, (-1, self.output_dim))
        value = np.reshape(value, (-1, 1))

        # Reserve samples for validation.
        val_samples = int(0.2 * states.shape[0])

        s_val = states[-val_samples:]
        p_val = policy[-val_samples:]
        v_val = value[-val_samples:]
        s_train = states[:-val_samples]
        p_train = policy[:-val_samples]
        v_train = value[:-val_samples]

        history = {"loss": [],
                   "val_loss": [],
                   "probabilities_loss": [],
                   "value_loss": [],
                   "val_probabilities_loss": [],
                   "val_value_loss": []}
        for epoch in range(epochs):
            # print("\nStart of epoch %d" % (epoch,))
            # Iterate over the batches of the dataset.
            avg_loss = total_loss = 0
            avg_pol_loss = total_pol_loss = 0
            avg_value_loss = total_value_loss = 0
            step = 0
            base = 0
            s_train, p_train, v_train = unison_shuffled(s_train, p_train, v_train)
            while base < s_train.shape[0] - batch_size:
                base = batch_size * step

                loss, policy_loss, value_loss = self._fit(s_train[base: base + batch_size],
                                                          p_train[base: base + batch_size],
                                                          v_train[base: base + batch_size])
                if np.isnan(loss):
                    print("What")
                total_loss += loss
                avg_loss = total_loss / (step + 1)

                total_pol_loss += policy_loss
                avg_pol_loss = total_pol_loss / (step + 1)

                total_value_loss += value_loss
                avg_value_loss = total_value_loss / (step + 1)
                step += 1
                # print("Epoch:", epoch, "; step:", step + 1, "loss:", loss)
            avg_loss_val = total_loss = 0
            avg_pol_loss_val = total_pol_loss = 0
            avg_value_loss_val = total_value_loss = 0
            step = 0
            base = 0
            while base < s_val.shape[0] - batch_size:
                base = batch_size * step
                loss, policy_loss, value_loss = self._evaluate(s_val[base: base + batch_size],
                                                                p_val[base: base + batch_size],
                                                                v_val[base: base + batch_size])
                total_loss += loss
                avg_loss_val = total_loss / (step + 1)

                total_pol_loss += policy_loss
                avg_pol_loss_val = total_pol_loss / (step + 1)

                total_value_loss += value_loss
                avg_value_loss_val = total_value_loss / (step + 1)
                step += 1
            history['loss'].append(avg_loss)
            history['val_loss'].append(avg_loss_val)
            history['probabilities_loss'].append(avg_pol_loss)
            history['value_loss'].append(avg_value_loss)
            history['val_probabilities_loss'].append(avg_pol_loss_val)
            history['val_value_loss'].append(avg_value_loss_val)
            if verbose > 0:
                print("Epoch ", epoch, ": loss:", avg_loss, "; val_los:", avg_loss_val, "; pol_loss:", avg_pol_loss,
                      "; pol_val_loss:", avg_pol_loss_val, "; value_loss:", avg_value_loss,
                      "; val_value_loss:", avg_value_loss_val)
        return History(history)
    
    def predict(self, x):
        P, v = self._predict(x)
        return P, v
    
    def predict_one(self, x):
        P, v = self.model._predict(x[None, ...])
        return P[0], v[0][0]
        
    def set_weights(self, weights):
        self.set_from_flat(weights)

    def get_weights(self,):
        sess = self.sess or tf.get_default_session()
        layers = sess.run(self.vars)
        weights = []
        for layer in layers:
            weights.append(layer.ravel())
        return np.concatenate(weights)

    def save(self, path):
        U.save_variables(path)

    def load(self, load_path):
        U.load_variables(load_path, variables=self.vars)

    def load_weights(self, load_path):
        U.load_variables(load_path, variables=self.vars)