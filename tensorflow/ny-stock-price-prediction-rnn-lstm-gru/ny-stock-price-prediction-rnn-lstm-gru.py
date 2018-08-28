# https://www.kaggle.com/raoulma/ny-stock-price-prediction-rnn-lstm-gru/notebook
# https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py

import argparse
import sys

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing

LEN_SEQUENCE = 20
LEN_OHLC = 4

df = pd.read_csv("./prices-split-adjusted.csv", index_col=0)

print(df.describe())

df_stock = df[df.symbol == 'EQIX'].copy().drop(['symbol', 'volume'], 1)
columns = list(df_stock.columns.values)

scaler = preprocessing.MinMaxScaler()

df_stock['open'] = scaler.fit_transform(df_stock['open'].values.reshape(-1, 1))
df_stock['high'] = scaler.fit_transform(df_stock['high'].values.reshape(-1, 1))
df_stock['low'] = scaler.fit_transform(df_stock['low'].values.reshape(-1, 1))
df_stock['close'] = scaler.fit_transform(df_stock['close'].values.reshape(-1, 1))

stock_values = df_stock.values
stock_sequences = []

for index in range(len(stock_values) - LEN_SEQUENCE):
    stock_sequences.append(stock_values[index:index + LEN_SEQUENCE])

stock_sequences = np.array(stock_sequences)

training_range = (0, round(stock_sequences.shape[0] * 0.7))
validating_range = (training_range[1], stock_sequences.shape[0])

training_features = stock_sequences[training_range[0]:training_range[1], :-1, :]
training_labels = stock_sequences[training_range[0]:training_range[1], -1, :]

validating_features = stock_sequences[validating_range[0]:validating_range[1], :-1, :]
validating_labels = stock_sequences[validating_range[0]:validating_range[1], -1, :]


def get_input_fn(mode, batch_size):
    """Creates an input_fn that stores all the data in memory.
    Args:
     mode: one of tf.contrib.learn.ModeKeys.{TRAIN, INFER, EVAL}
     batch_size: the batch size to output.
    Returns:
      A valid input_fn for the model estimator.
    """

    def _input_fn():
        """Estimator `input_fn`.
        Returns:
          A tuple of:
          - Dictionary of string feature name to `Tensor`.
          - `Tensor` of target labels.
        """
        if mode == tf.estimator.ModeKeys.TRAIN:
            features, labels = training_features, training_labels
        else:
            features, labels = validating_features, validating_labels

        features = {"ohlc": features.astype(np.float32)}
        labels = {"ohlc": labels.astype(np.float32)}

        dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)

        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.repeat(None).shuffle(buffer_size=training_features.shape[0])
        else:
            dataset = dataset.repeat(1)

        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels

    return _input_fn


def model_fn(features, labels, mode, params):
    """Model function for RNN classifier.
    Args:
      features: dictionary with keys: inks, lengths.
      labels: one hot encoded classes
      mode: one of tf.estimator.ModeKeys.{TRAIN, INFER, EVAL}
      params: a parameter dictionary with the following keys: num_layers,
        num_nodes, batch_size, num_classes, learning_rate.
    Returns:
      ModelFnOps for Estimator API.
    """

    layers = [tf.contrib.rnn.BasicLSTMCell(num_units=params.num_nodes, activation=tf.nn.elu)
              for _ in range(params.num_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)

    # outputs is [batch_size, LEN_SEQUENCE - 1, params.num_nodes]
    outputs, _ = tf.nn.dynamic_rnn(multi_layer_cell, features["ohlc"], dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(outputs, [-1, params.num_nodes])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, LEN_OHLC)

    outputs = tf.reshape(stacked_outputs, [-1, LEN_SEQUENCE - 1, LEN_OHLC])
    outputs = outputs[:, LEN_SEQUENCE - 2, :]

    # Add the loss.
    loss = tf.losses.mean_squared_error(labels["ohlc"], outputs)
    rmse = tf.metrics.root_mean_squared_error(labels["ohlc"], outputs)

    # Add the optimizer.
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=params.learning_rate,
        optimizer="Adam",
        # some gradient clipping stabilizes training in the beginning.
        clip_gradients=params.gradient_clipping_norm,
        summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"predictions": outputs},
        loss=loss,
        train_op=train_op,
        eval_metric_ops={"rmse": rmse})


def create_estimator_and_specs(run_config):
    """Creates an Experiment configuration based on the estimator and input fn."""
    model_params = tf.contrib.training.HParams(
        num_layers=FLAGS.num_layers,
        num_nodes=FLAGS.num_nodes,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        gradient_clipping_norm=FLAGS.gradient_clipping_norm)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=model_params)

    train_spec = tf.estimator.TrainSpec(input_fn=get_input_fn(
        mode=tf.estimator.ModeKeys.TRAIN,
        batch_size=FLAGS.batch_size), max_steps=FLAGS.steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=get_input_fn(
        mode=tf.estimator.ModeKeys.EVAL,
        batch_size=FLAGS.batch_size))

    return estimator, train_spec, eval_spec


def main(args):
    estimator, train_spec, eval_spec = create_estimator_and_specs(
        run_config=tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir,
            save_checkpoints_secs=300,
            save_summary_steps=100))
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of recurrent neural network layers.")
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=200,
        help="Number of node per recurrent network layer.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate used for training.")
    parser.add_argument(
        "--gradient_clipping_norm",
        type=float,
        default=9.0,
        help="Gradient clipping norm used during training.")
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of training steps.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size to use for training/evaluation.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./output",
        help="Path for storing the model checkpoints.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
