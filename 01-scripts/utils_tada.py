import sys
sys.path.insert(1, '../01-scripts/05-tada')

import os
import pandas as pd
import numpy as np
from Model import create_model, attention, f1_metric
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from sklearn.utils import class_weight
from typing import List, Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Bidirectional, LSTM, Dense, GlobalMaxPooling1D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow as tf
from loss import focal_loss

from Preprocessing import create_features, scale_features
from utils_data import save_pickle

np.random.seed(0)

DataSplit = Tuple[np.ndarray, np.ndarray, np.ndarray]

def preprocess_TADA(X: List,
                    y: np.ndarray,
                    idxs: Tuple,
                    feature_file: str,
                    seq_window: int = 5,
                    steps: int = 1,
                    seq_len: int = 40
) -> Tuple[DataSplit, DataSplit, DataSplit]:
    
    train_idx, val_idx, test_idx = idxs
    if not os.path.isfile(feature_file):
        X = create_features(X,seq_window,steps,seq_len)
        save_pickle(X, feature_file)
    else:
        X = pd.read_pickle(feature_file)
    X = scale_features(X_train)

    X_train = X[train_idx,:]
    X_val = X[val_idx,:]
    X_test = X[test_idx,:]

    y = (y > 1).astype(int)
    y = np.column_stack([y, 1 - y])
    y_train = y[train_idx,:]
    y_val = y[val_idx,:]
    y_test = y[test_idx,:]

    train = (X_train, y_train)
    val =  (X_val, y_val)
    test = (X_test, y_test)
    return (train, val, test)
    
def train_model(dataset: Tuple[DataSplit, DataSplit, DataSplit],
          model: Sequential, 
          out_dir: str,
          out_file: str
) -> None:
    (X_train_scaled, y_train, X_val_scaled, y_val, _, _ ) = dataset

    model_checkpoint = ModelCheckpoint(filepath=f'{out_dir}/{out_file}_checkpoints'+'/tada.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       save_weights_only=True,
                                       monitor='val_loss',
                                       mode='auto',
                                       save_best_only=True)
    history = History()
    earlystopper = EarlyStopping(monitor='val_f1_metric',patience=7,verbose=1)
    callbacks = [history,model_checkpoint,earlystopper]
    ground_truth = np.argmax(y_train, axis=-1)
    class_weights = class_weight.compute_class_weight('balanced',classes=np.unique(ground_truth),y=ground_truth)
    d_class_weights = dict(enumerate(class_weights))
    history = model.fit(X_train_scaled,
                        y_train,
                        batch_size=64,
                        epochs=100,
                        verbose=1,
                        callbacks=callbacks,
                        class_weight = d_class_weights,
                        validation_data=(X_val_scaled, y_val))
    train_df = pd.DataFrame(history.history)
    train_df.to_pickle(f'{out_dir}/{out_file}_training.pkl')
    model.save(f'{out_dir}/{out_file}_tada.h5')

def test_model(dataset: Tuple[DataSplit, DataSplit, DataSplit],
         model: Sequential,
         out_dir: str,
         out_file: str
) -> None:
    (_,_,_,_,X_test_scaled,y_test) = dataset
    
    results = model.evaluate(X_test_scaled,y_test,verbose = 1)
    results_df = pd.DataFrame(results).T
    results_df.columns = ['loss','precision','recall','auc','accuracy','aupr','f1_metric']
    results_df.to_pickle(f'{out_dir}/{out_file}_test_results.pkl')
    
    predictions = model.predict(X_test_scaled)
    y_test_hat = predictions[:,0]
    y_test_hat_bin = np.where(y_test_hat>0.4,1,0)

    test_df = pd.DataFrame([y_test[:,0], y_test_hat, y_test_hat_bin]).T
    test_df.columns = ["y_test", "y_test_hat","y_test_hat_bin"]
    test_df.to_pickle(f'{out_dir}/{out_file}_test_data.pkl')

def evaluate_TADA(X: List[str],
                  y: np.ndarray,
                  idxs: DataSplit,
                  feature_file: str,
                  out_dir: str,
                  out_file: str,
                  seq_window: int = 5,
                  steps: int = 1,
                  seq_len: int = 40,
                  random_state: int = 0
) -> None:
    np.random.seed(random_state)

    dataset = preprocess_TADA(X,y,idxs,feature_file,seq_window,steps,seq_len)

    input_shape = X[0].shape
    model = create_model(input_shape)
    model.summary()
    train_model(dataset,model,out_dir,out_file)

    model = create_model(input_shape)
    model.load_weights(f'{out_dir}/{out_file}_tada.h5')
    test_model(dataset,model,out_dir,out_file)

def ablate_TADA(input_shape,
                kernel_size=2,
                filters=100,
                activation_function='gelu',
                learning_rate=1e-3,
                dropout=0.3,
                bilstm_output_size=100):
    
    def create_model_variant(layers_fn,alpha=0.45):
        metric = [tf.keras.metrics.Precision(name = 'precision'),
        tf.keras.metrics.Recall(name = 'recall'), 
        tf.keras.metrics.AUC(name = 'auc', curve = 'ROC'),
        tf.keras.metrics.CategoricalAccuracy(name ='accuracy'),
        tf.keras.metrics.AUC(name = 'aupr', curve = 'PR'),
        f1_metric
        ]
        model = Sequential()
        layers_fn(model)
        model.add(Dense(2,activation="softmax"))
        opt = Adam(learning_rate)
        model.compile(loss=focal_loss(alpha=alpha),optimizer=opt,metrics=metric)
        return model

    ablations = {}

    def full_model(model):
        model.add(Conv1D(filters=filters,
                         kernel_size=kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1,
                         input_shape=input_shape,
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        model.add(Dropout(dropout))
        model.add(Conv1D(filters=filters,
                         kernel_size=kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1))
        model.add(Dropout(dropout))
        model.add(attention())
        model.add(Bidirectional(LSTM(bilstm_output_size,return_sequences=True)))
        model.add(Bidirectional(LSTM(bilstm_output_size)))
    ablations["Conv1D-Dropout-Conv1D-Dropout-Attention-BiLSTM-BiLSTM"] = create_model_variant(full_model)

    def no_attention(model):
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1,
                         input_shape=input_shape,
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        model.add(Dropout(dropout))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(bilstm_output_size,return_sequences=True)))
        model.add(Bidirectional(LSTM(bilstm_output_size)))
    ablations["Conv1D-Dropout-Conv1D-Dropout-BiLSTM-BiLSTM"] = create_model_variant(no_attention)

    def no_dropout(model):
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1,
                         input_shape=input_shape,
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1))
        model.add(attention())
        model.add(Bidirectional(LSTM(bilstm_output_size, return_sequences=True)))
        model.add(Bidirectional(LSTM(bilstm_output_size)))
    ablations["Conv1D-Conv1D-Attention-BiLSTM-BiLSTM"] = create_model_variant(no_dropout)

    def no_dropout_attention(model):
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1,
                         input_shape=input_shape,
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1))
        model.add(Bidirectional(LSTM(bilstm_output_size, return_sequences=True)))
        model.add(Bidirectional(LSTM(bilstm_output_size)))
    ablations["Conv1D-Conv1D-BiLSTM-BiLSTM"] = create_model_variant(no_dropout_attention)
    
    def single_lstm(model):
        model.add(Conv1D(filters,
                         kernel_size, 
                         padding='valid',
                         activation=activation_function,
                         strides=1,
                         input_shape=input_shape,
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        model.add(Dropout(dropout))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1))
        model.add(Dropout(dropout))
        model.add(attention())
        model.add(Bidirectional(LSTM(bilstm_output_size)))
    ablations["Conv1D-Dropout-Conv1D-Dropout-Attention-BiLSTM"] = create_model_variant(single_lstm)

    def no_lstm(model):
        model.add(Conv1D(filters,
                         kernel_size, 
                         padding='valid',
                         activation=activation_function,
                         strides=1,
                         input_shape=input_shape,
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        model.add(Dropout(dropout))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1))
        model.add(Dropout(dropout))
        model.add(attention())
        model.add(GlobalMaxPooling1D())
    ablations["Conv1D-Dropout-Conv1D-Dropout-Attention-MaxPooling1D"] = create_model_variant(no_lstm)

    def no_lstm_attention(model):
        model.add(Conv1D(filters,
                         kernel_size, 
                         padding='valid',
                         activation=activation_function,
                         strides=1,
                         input_shape=input_shape,
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        model.add(Dropout(dropout))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1))
        model.add(Dropout(dropout))
        model.add(GlobalMaxPooling1D())
    ablations["Conv1D-Dropout-Conv1D-Dropout-MaxPooling1D"] = create_model_variant(no_lstm_attention)

    def only_cnn(model):
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1,
                         input_shape=input_shape,
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1))
        model.add(GlobalMaxPooling1D())
    ablations["Conv1D-Conv1D-MaxPooling1D"] = create_model_variant(only_cnn)
    
    def single_cnn(model):
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1,
                         input_shape=input_shape,
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        model.add(GlobalMaxPooling1D())
    ablations["Conv1D-MaxPooling1D"] = create_model_variant(single_cnn)

    def single_cnn_dropout(model):
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation=activation_function,
                         strides=1,
                         input_shape=input_shape,
                         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
        model.add(Dropout(dropout))
        model.add(GlobalMaxPooling1D())
    ablations["Conv1D-Dropout-MaxPooling1D"] = create_model_variant(single_cnn_dropout)

    def only_attention(model):
        model.add(attention())
        model.add(GlobalMaxPooling1D())
    ablations["Attention-MaxPooling1D"] = create_model_variant(only_attention)

    def only_lstm(model):
        model.add(Bidirectional(LSTM(bilstm_output_size)))
    ablations["BiLSTM"] = create_model_variant(only_lstm)

    def minimal_fcnn(model):
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(64, activation=activation_function))
    ablations["Flatten-Dense"] = create_model_variant(minimal_fcnn)

    def only_flat(model):
        model.add(Flatten(input_shape=input_shape))
    ablations["Flatten"] = create_model_variant(only_flat)

    return ablations

def evaluate_ablation(dataset,
                     input_shape,
                     out_dir,
                     random_state=0,
                     kernel_size=2,
                     filters=100,
                     activation_function='gelu',
                     learning_rate=1e-3,
                     dropout=0.3,
                     bilstm_output_size=100,
) -> None:
    
    ablation_models = ablate_TADA(input_shape,
                kernel_size,
                filters,
                activation_function,
                learning_rate,
                dropout,
                bilstm_output_size)
    
    for name, model in ablation_models.items():
        out_file = f'ablation_{name}_{random_state}'
        if not os.path.isdir(f'{out_dir}/{out_file}_checkpoints'):
            os.makedirs(f'{out_dir}/{out_file}_checkpoints',exist_ok=True)
            train_model(dataset,model,out_dir,out_file)
            test_model(dataset,model,out_dir,out_file)
    return ablation_models