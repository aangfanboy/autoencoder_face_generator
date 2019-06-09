import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, data_dir: str, img_shape: tuple = (48, 48, 1)):
        self.data_dir = data_dir
        self.xs, self.ys, self.channels = img_shape

    def load_images(self):
        try:
            x_data = np.load("x_data.npy")
        except FileNotFoundError:
            x_data = []

            for img_p in tqdm(os.listdir(self.data_dir), "Loading images..."):
                if img_p.endswith(".jpg"):
                    img = cv2.imread(os.path.join(self.data_dir, img_p))
                    img = cv2.resize(img, (self.xs, self.ys))

                    if self.channels == 1:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    x_data.append(img)

            x_data = np.array(x_data)
            np.save("x_data.npy", x_data)

        if self.channels == 1:
            x_data = np.expand_dims(x_data, axis=-1)
        return x_data


class Network:
    def __init__(self, image_placeholder, bottleneck_placeholder, img_shape: tuple = (48, 48, 1)):
        self.xs, self.ys, self.channels = img_shape
        self.bottleneck_placeholder = bottleneck_placeholder
        self.image_pc = image_placeholder

        self.bottleneck_size = bottleneck_placeholder.shape[-1]

        self.encoder = self.get_encoder(l2_on_last_layer=True)
        self.decoder = self.get_decoder(self.bottleneck_placeholder)

    def get_encoder(self, l2_on_last_layer: bool = True):
        with tf.variable_scope("encoder"):
            x = tf.layers.conv2d(self.image_pc, 64, (3, 3), strides=2, activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(5e-4))
            x = tf.layers.conv2d(x, 128, (3, 3), strides=1, activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(5e-4))
            x = tf.layers.conv2d(x, 256, (3, 3), strides=2, activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(5e-4))
            x = tf.layers.max_pooling2d(x, (3, 3), strides=2)
            x = tf.layers.conv2d(x, 256, (3, 3), strides=1, activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(5e-4))

            x = tf.layers.average_pooling2d(x, (1, 1), strides=1)
            x = tf.layers.flatten(x)

            if l2_on_last_layer:
                x = tf.layers.dense(x, 128, activation=tf.nn.tanh, kernel_regularizer=tf.keras.regularizers.l2(5e-4))
            else:
                x = tf.layers.dense(x, 128, activation=tf.nn.tanh)

            return x

    def get_decoder(self, bl_pc):
        with tf.variable_scope("decoder"):
            x = tf.layers.dense(bl_pc, 144, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(5e-4))
            x = tf.layers.dense(x, 169, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(5e-4))
            x = tf.reshape(x, (-1, 13, 13, 1))

            x = tf.layers.conv2d_transpose(x, 64, (3, 3), strides=1, activation=tf.nn.relu, padding="SAME",
                                           kernel_regularizer=tf.keras.regularizers.l2(5e-4))
            x = tf.layers.conv2d(x, 64, (2, 2), strides=1, activation=tf.nn.relu)
            x = tf.layers.batch_normalization(x, momentum=0.8)
            x = tf.layers.conv2d_transpose(x, 128, (3, 3), strides=1, activation=tf.nn.relu, padding="SAME",
                                           kernel_regularizer=tf.keras.regularizers.l2(5e-4))
            x = tf.layers.conv2d_transpose(x, 128, (3, 3), strides=2, activation=tf.nn.relu, padding="SAME",
                                           kernel_regularizer=tf.keras.regularizers.l2(5e-4))
            x = tf.layers.conv2d_transpose(x, 256, (3, 3), strides=2, activation=tf.nn.relu, padding="SAME",
                                           kernel_regularizer=tf.keras.regularizers.l2(5e-4))
            x = tf.layers.conv2d_transpose(x, self.channels, (3, 3), strides=1, activation=None, padding="SAME")

        return x


class TrainAutoEncoderModel:
    def __init__(self, x_data, img_shape: tuple = (48, 48, 1), decay_steps: int = 250, lr: float = 0.001,
                 model_dir: str = "models/", epochs: int = 15, batch_size: int = 128, dir2save="predicted_images/"):
        self.epochs, self.batch_size, = epochs, batch_size
        self.xs, self.ys, self.channels = img_shape
        self.x_data = x_data / 255.0
        self.dir2save = dir2save
        self.decay_steps = decay_steps
        self.model_dir = model_dir
        self.lr = lr
        self.x_train, self.x_test, _, _ = train_test_split(self.x_data, np.zeros_like(self.x_data), test_size=0.01,
                                                           random_state=42, shuffle=True)
        tf.logging.info(" Model splitted up!")

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        os.makedirs(os.path.dirname(self.dir2save), exist_ok=True)

        self.image_pc = tf.placeholder(tf.float32, [None, self.xs, self.ys, self.channels], name="image_pc")
        self.bottleneck_pc = tf.placeholder(tf.float32, [None, 128], name="bottleneck_pc")
        self.global_step = tf.Variable(0., name="global_step")
        tf.logging.info(" Placeholders created!")

        self.nw = Network(self.image_pc, self.bottleneck_pc, img_shape=img_shape)
        self.encoder = self.nw.encoder
        self.decoder = self.nw.decoder
        tf.logging.info(" Model created!")

        self.loss = tf.losses.mean_squared_error(self.image_pc, self.decoder)
        tf.logging.info(" Loss set!")

        self.train_s = tf.summary.merge([tf.summary.scalar("loss", self.loss)])
        self.test_s = tf.summary.merge([tf.summary.scalar("val_loss", self.loss)])
        tf.logging.info(" TensorBoard set!")

        if self.decay_steps is not None and self.decay_steps > 0:
            self.lr = tf.train.exponential_decay(self.lr, self.global_step, self.decay_steps, 0.96, staircase=True)
        else:
            self.lr = tf.Variable(self.lr, trainable=False)

        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, self.global_step, tf.trainable_variables())
        tf.logging.info(" Optimizer set!")
        self.saver = tf.train.Saver(tf.trainable_variables())
        tf.logging.info(" TensorFlow Graph Saver set!")
        tf.logging.info(" TrainAutoEncoderModel created!")

    def test_model(self, sess, file_writer=None, save_images: bool = True, n2d: int = 0):
        losses_t = []
        i = 0
        for epoch in range(self.epochs):
            q = int(self.x_test.shape[0] / self.batch_size)
            for batch in range(q):
                images_batch = self.x_test[batch * self.batch_size: (batch + 1) * self.batch_size]

                encoder_output = sess.run(self.encoder, feed_dict={
                    self.image_pc: images_batch,
                })

                pre_img, _, test_loss, cnt, test_summ = sess.run([self.decoder, self.opt, self.loss, self.global_step,
                                                                  self.test_s],
                                                                 feed_dict={
                                                                  self.bottleneck_pc: encoder_output,
                                                                  self.image_pc: images_batch,
                                                                    })

                if file_writer is not None:
                    file_writer.add_summary(test_summ, cnt)

                if save_images:
                    cv2.imwrite(os.path.join(f"{self.dir2save}/{n2d}_{i}.jpg"),
                                pre_img[np.random.choice(len(pre_img), 1)].reshape(self.xs, self.ys, self.channels)*255)
                    i += 1

                losses_t.append(test_loss)

        return round(float(np.mean(losses_t)), 5)

    def train_model(self):
        losses = []
        with tf.Session() as sess:
            tf.logging.info(" Session created!")
            sess.run(tf.global_variables_initializer())
            tf.logging.info(" Initialized!")

            file_writer = tf.summary.FileWriter("autoencoder_graphs", sess.graph)
            tf.logging.info(" TensorBoard File Writer created! Writing on 'autoencoder_graphs/'")

            try:
                self.saver.restore(sess, os.path.join(self.model_dir, "tf_model.ckpt"))
                tf.logging.info(" Model Restored!")
            except ValueError:
                tf.logging.info(" No model to restore!")

            for epoch in range(self.epochs):
                q = int(self.x_train.shape[0] / self.batch_size)
                bar = tqdm(total=q)
                for batch in range(q):
                    images_batch = self.x_train[batch * self.batch_size: (batch + 1) * self.batch_size]

                    encoder_output = sess.run(self.encoder, feed_dict={
                        self.image_pc: images_batch,
                    })

                    _, loss_v, cnt, train_summ, lr_v = sess.run([self.opt, self.loss, self.global_step, self.train_s,
                                                                 self.lr], feed_dict={
                                                                      self.bottleneck_pc: encoder_output,
                                                                      self.image_pc: images_batch,
                                                                                        })

                    file_writer.add_summary(train_summ, cnt)
                    losses.append(loss_v)
                    if len(losses) == 25:
                        losses.pop(0)

                    bar.update()
                    bar.set_description(f"{epoch}/{cnt} || Lr --> {round(float(lr_v), 5)} || "
                                        f"Loss --> {round(float(np.mean(losses)), 5)}")

                test_loss = self.test_model(sess, file_writer=file_writer, n2d=epoch)
                statue = True

                if test_loss - float(np.mean(losses)) > 150.0:
                    statue = False

                bar.set_description(f"{statue}! || #{epoch}/{int(cnt)} || Lr --> {round(float(lr_v), 5)} || "
                                    f"Loss --> {round(float(np.mean(losses)), 5)} || "
                                    f"Test Loss --> {test_loss}")
                bar.close()
                if int(epoch) % 5 == 0:
                    tf.logging.info(" Model is saving...")
                    self.saver.save(sess, f"{self.model_dir}/tf_model.ckpt")
                    tf.logging.info(" Model saved!")

            tf.logging.info(" Model is saving...")
            self.saver.save(sess, f"{self.model_dir}/tf_model.ckpt")
            tf.logging.info(" Model saved!")
            sess.close()
            tf.logging.info(" Session Closed!")

        tf.logging.info(" Training is done!")

    def give_examples(self):
        n = 144
        with tf.Session() as sess:
            tf.logging.info(" Session created!")
            sess.run(tf.global_variables_initializer())
            tf.logging.info(" Initialized!")

            try:
                self.saver.restore(sess, os.path.join(self.model_dir, "tf_model.ckpt"))
                tf.logging.info(" Model Restored!")
            except ValueError:
                tf.logging.info(" No model to restore!")

            images = np.random.choice(len(self.x_data), n)
            images = self.x_data[images]

            encoder_output = sess.run(self.encoder, feed_dict={
                self.image_pc: images,
            })

            decoded_real = sess.run(self.decoder, feed_dict={self.bottleneck_pc: encoder_output,
                                                             self.image_pc: images})

            a = np.sqrt(n)
            fig = plt.figure(figsize=(a, a))
            ax = plt.gca()
            ax.set_facecolor('xkcd:black')
            for i, img in enumerate(decoded_real):
                fig.add_subplot(a, a, i+1)
                plt.axis("off")
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.savefig("fig.png")


if __name__ == '__main__': 
    dl = DataLoader("dataset", (48, 48, 3))
    X_data = dl.load_images()

    trainer = TrainAutoEncoderModel(X_data, (48, 48, 3), decay_steps=200, lr=0.0008, model_dir="autoencoder_model/")
    trainer.give_examples()
