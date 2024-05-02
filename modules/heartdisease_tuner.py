from typing import NamedTuple, Dict, Any

import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from keras import layers
from keras_tuner.engine import base_tuner

from heartdisease_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)

TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', base_tuner.BaseTuner),
    ('fit_kwargs', Dict[str, Any]),
])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_binary_accuracy',
    mode='max',
    verbose=1,
    patience=5
)

def gzip_reader_fn(filenames: str) -> tf.data.TFRecordDataset:
    """Load data from GZIP-compressed TFRecords."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern: str, tf_transform_output: tft.TFTransformOutput, 
             batch_size: int = 64) -> tf.data.Dataset:
    """Generates features and labels for tuning/training."""
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset

def get_tuner_model(hyperparameters: kt.HyperParameters, 
                    show_summary: bool = True) -> tf.keras.Model:
    """Define a Keras model for tuning."""
    num_layers = hyperparameters.Choice('num_layers', values=[1, 2])
    dense_units = hyperparameters.Int('dense_units', min_value=16, max_value=256, step=64)
    dropout_rate = hyperparameters.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    learning_rate = hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3])

    inputs = [layers.Input(shape=(dim+1,), name=transformed_name(key))
              for key, dim in CATEGORICAL_FEATURES.items()]
    inputs.extend(layers.Input(shape=(1,), name=transformed_name(feature))
                  for feature in NUMERICAL_FEATURES)

    concatenated = layers.concatenate(inputs)
    x = layers.Dense(dense_units, activation='relu')(concatenated)

    for _ in range(num_layers - 1):
        x = layers.Dense(dense_units, activation='relu')(x)

    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['binary_accuracy']
    )

    if show_summary:
        model.summary()

    return model

def tuner_fn(fn_args: Any) -> TunerFnResult:
    """Tune the model to find the best hyperparameters."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files[0], tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files[0], tf_transform_output)

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: get_tuner_model(hp),
        objective=kt.Objective('val_loss', direction='min'),
        max_trials=2,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='heartdisease_kt'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
            'callbacks': [early_stop]
        }
    )
