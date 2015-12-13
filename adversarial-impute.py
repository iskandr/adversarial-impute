
from keras.models import Sequential, Graph
from keras.objectives import mse
from keras.layers.core import Dropout, Dense
import numpy as np
import theano

import seaborn


def create_models(
        n_dims=25,
        dropout_probability=0.1,
        adversarial_weight=1,
        reconstruction_weight=1,
        generator_hidden_size_multipliers=[4, 0.5],
        decoder_hidden_size_multiplier=[4, 2, 1, 0.5],
        activation="relu"):
    decoder = Graph()
    decoder.add_input(name="input", input_shape=(n_dims,))
    last_layer = "input"
    for i, size in enumerate(decoder_hidden_size_multiplier):
        hidden_name = "dense%d" % (i + 1)
        decoder.add_node(
            layer=Dense(size * n_dims, activation=activation),
            name=hidden_name,
            inputs=[last_layer, "input"])
        dropout_name = hidden_name + "_dropout"
        decoder.add_node(
            layer=Dropout(dropout_probability),
            name=dropout_name,
            input=hidden_name)
        last_layer = dropout_name
    decoder.add_node(
        layer=Dense(n_dims, activation="sigmoid"),
        name="output",
        inputs=[last_layer, "input"],
        create_output=True)
    decoder.compile(optimizer="rmsprop", loss={"output": "binary_crossentropy"})

    def generator_loss(combined, imputed_vector):
        """
        Ignores y_true and y_pred
        """
        original_vector = combined[:, :n_dims]
        missing_mask = combined[:, n_dims:]
        input_variable = decoder.get_input()
        decoder_compute_graph = decoder.get_output()
        mask_prediction = theano.clone(
            decoder_compute_graph,
            {input_variable: imputed_vector},
            share_inputs=True)
        reconstruction_loss = mse(
            y_true=original_vector * (1 - missing_mask),
            y_pred=imputed_vector * (1 - missing_mask))
        decoder_mask_loss = mse(missing_mask, missing_mask * mask_prediction)
        return (
            reconstruction_weight * reconstruction_loss
            - adversarial_weight * decoder_mask_loss)

    generator = Sequential()
    generator.add(Dense(
        (generator_hidden_size_multipliers[0] * n_dims),
        input_dim=2 * n_dims,
        activation=activation))
    generator.add(Dropout(dropout_probability))

    for layer_size_multiplier in generator_hidden_size_multipliers[1:] + [1]:
        generator.add(Dense(
            int(layer_size_multiplier * n_dims),
            activation=activation))
        generator.add(Dropout(dropout_probability))
    generator.add(Dense(n_dims, activation='linear'))
    generator.compile(optimizer="rmsprop", loss=generator_loss)
    return generator, decoder


def create_data(
        n_samples=1000,
        n_dims=25,
        offset=0,
        fraction_missing=0.5):
    t = np.linspace(-1, 1, n_dims, endpoint=False)
    assert len(t) == n_dims

    X_full = np.zeros((n_samples, n_dims))

    for i in range(n_samples):
        phase = np.random.randn() * np.pi / 2
        frequency = 10 * np.random.rand()
        X_full[i, :] = offset + np.sin(phase + t * frequency)
    missing_mask = np.random.random(X_full.shape) < fraction_missing
    X_incomplete = X_full.copy()
    X_incomplete[missing_mask] = 0.0
    return X_full, X_incomplete, missing_mask


def pretrain_decoder(
        X_incomplete,
        missing_mask,
        decoder,
        batch_size=128,
        training_epochs=5):
    X_incomplete = X_incomplete.copy()
    n_samples, n_dims = X_incomplete.shape
    X_incomplete[missing_mask] = np.nan
    feature_means = np.nanmean(X_incomplete, axis=0)
    assert len(feature_means) == n_dims
    feature_stds = np.nanstd(X_incomplete, axis=0)
    assert len(feature_stds) == n_dims
    X_incomplete[missing_mask] = 0.0

    indices = np.arange(n_samples)
    for epoch in range(training_epochs):
        np.random.shuffle(indices)
        n_batches = n_samples // batch_size
        for batch_idx in range(n_batches):
            batch_indices = indices[
                batch_idx * batch_size:(batch_idx + 1) * batch_size]
            X_batch = X_incomplete[batch_indices]
            missing_mask_batch = missing_mask[batch_indices]
            X_batch_noisy = X_batch.copy()
            for feature_idx in range(n_dims):
                noise = np.random.randn(batch_size) * feature_stds[feature_idx] + feature_means[feature_idx]
                missing_rows = missing_mask_batch[:, feature_idx]
                X_batch_noisy[missing_rows, feature_idx] = noise[missing_rows]

            predicted_mask = decoder.predict({"input": X_batch_noisy})["output"]
            print("Pre-training epoch %d, mini-batch %d, accuracy=%0.4f" % (
                epoch + 1,
                batch_idx + 1,
                ((predicted_mask > 0.5) == missing_mask_batch).mean(),))
            decoder.train_on_batch({
                "input": X_batch_noisy,
                "output": missing_mask_batch
            })


def train(
        X_full,
        X_incomplete,
        mask,
        generator,
        decoder,
        batch_size=128,
        training_epochs=10,
        alternating_updates=False,
        plot_each_epoch=False,
        decoder_pretrain_epochs=5):
    combined = np.hstack([X_incomplete, mask])

    n_samples = len(X_full)
    indices = np.arange(n_samples)

    for epoch in range(training_epochs):
        np.random.shuffle(indices)
        n_batches = n_samples // batch_size
        for batch_idx in range(n_batches):
            batch_indices = indices[
                batch_idx * batch_size:(batch_idx + 1) * batch_size]
            combined_batch = combined[batch_indices]
            mask_batch = mask[batch_indices]
            X_imputed = generator.predict(combined_batch)
            X_full_batch = X_full[batch_indices]

            reconstruction_mse = ((X_imputed - X_full_batch) ** 2).mean()
            predicted_mask = decoder.predict({"input": X_imputed})["output"]
            masking_mse = ((mask_batch - predicted_mask) ** 2).mean()
            if np.isnan(reconstruction_mse):
                raise ValueError("Generator Diverged!")
            if np.isnan(masking_mse):
                raise ValueError("Decoder Diverged!")
            print((
                "-- Epoch %d, batch %d, "
                "Reconstruction MSE = %0.4f, "
                "Decoder MSE = %0.4f, "
                "Decoder accuracy = %0.4f "
                "(mean mask prediction = %0.4f)") % (
                epoch + 1,
                batch_idx + 1,
                reconstruction_mse,
                masking_mse,
                ((predicted_mask > 0.5) == mask_batch).mean(),
                predicted_mask.mean()))
            print("Decoder mask predictions: %s" % (
                list(zip(predicted_mask[0], mask_batch[0])),))
            if not alternating_updates or batch_idx % 2 == 0:
                decoder_input = X_imputed.copy()
                decoder_input[~mask_batch] = X_full_batch[~mask_batch]
                decoder.train_on_batch({
                    "input": decoder_input,
                    "output": mask_batch
                })
            if not alternating_updates or batch_idx % 2 == 1:
                generator.train_on_batch(
                    X=combined_batch,
                    y=combined_batch)
        if plot_each_epoch or epoch == training_epochs - 1:
            seaborn.plt.plot(X_imputed[0, :], label="X_imputed")
            seaborn.plt.plot(X_full_batch[0, :n_dims], label="X_full")
            seaborn.plt.plot(mask_batch[0], label="mask")
            seaborn.plt.legend()
            seaborn.plt.show()

if __name__ == "__main__":
    n_dims = 50
    n_samples = 10 ** 5
    training_epochs = 10
    batch_size = 128
    pretrain = False

    X_full, X_incomplete, missing_mask = create_data(
        n_dims=n_dims,
        n_samples=n_samples,
        fraction_missing=0.9)
    generator, decoder = create_models(
        n_dims=n_dims,
        activation="relu",
        adversarial_weight=0.1)

    if pretrain:
        pretrain_decoder(
            X_incomplete=X_incomplete,
            missing_mask=missing_mask,
            decoder=decoder,
            training_epochs=training_epochs,
            batch_size=batch_size)

    train(
        X_full,
        X_incomplete,
        missing_mask,
        generator=generator,
        decoder=decoder,
        training_epochs=10,
        alternating_updates=False,
        plot_each_epoch=False,
        batch_size=batch_size)
