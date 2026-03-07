// title: study.7-ann-ff
// date: 2026-02-26
// description: ANN baseline: feedforward ReLU network on MNIST as upper-bound for SNN studies.

#let config = json("_artifacts/study.7-ann-ff/config.json")
#let results = json("_artifacts/study.7-ann-ff/results.json")






= Summary


This study trains a conventional artificial neural network on MNIST to
establish a baseline for comparison with the spiking neural networks in
studies 8–10. The network is a three-layer feedforward MLP with ReLU
activations, trained end-to-end with Adam and cross-entropy loss. No
biophysics, no spike dynamics, no temporal simulation — just standard
deep learning.

The goal is to answer: how well does a simple ANN do on MNIST with the
same optimizer (Adam) and a comparable number of training epochs, so we
can put the SNN results in context?


= Architecture


#table(
  columns: 3,
  [Layer], [Size], [Activation],
  [Input], [784 (28x28 flattened)], [—],
  [Hidden 1], [512], [ReLU],
  [Hidden 2], [512], [ReLU],
  [Output], [10], [— (logits)],
)


This is a standard PyTorch `nn.Sequential` MLP. The two hidden layers have
512 neurons each — significantly larger than the 64-neuron hidden layer used
in the SNN studies (8–10), giving the ANN a clear capacity advantage.


= Config Snapshot


#table(
  columns: 2,
  [Key], [Value],
  [`meta.batch_size`], [#config.meta.batch_size],
  [`meta.epochs`], [#config.meta.epochs],
)



= Method



== Training


The full MNIST training set (#results.train_samples images) is used with a batch size of #config.meta.batch_size.
The optimizer is Adam with a learning rate of $10^(-3)$. Cross-entropy
loss is applied to the raw logits from the output layer.

No data augmentation, no learning rate schedule, no regularization. This is
deliberately minimal — the point is a quick baseline, not a tuned model.


== Evaluation


Test accuracy is computed on the full #results.test_samples-image MNIST test set at the end
of each epoch.


= Results



== Results Snapshot


#table(
  columns: 2,
  [Metric], [Value],
  [Best test accuracy], [#results.best_test_accuracy% (epoch #results.best_test_accuracy_epoch)],
  [Final test accuracy], [#results.final_test_accuracy%],
  [Best test loss], [#results.best_test_loss (epoch #results.best_test_loss_epoch)],
  [Final test loss], [#results.final_test_loss],
  [Trainable params], [#results.trainable_params],
)


// Gallery: loss*.png
// #figure(image("_artifacts/study.7-ann-ff/lossdark.png"), caption: [Test loss per epoch.])


Figure 1.1 shows the test loss declining across all #results.epochs epochs. The
initial loss is consistent with random 10-class prediction
($-ln(0.1) approx 2.3$). By epoch #results.epochs the loss is #results.final_test_loss, still dropping —
the network has not converged and would benefit from additional training.

// Gallery: accuracy*.png
// #figure(image("_artifacts/study.7-ann-ff/accuracydark.png"), caption: [Test accuracy per epoch.])


Figure 1.2 shows test accuracy rising to *#results.best_test_accuracy%*
after epoch #results.best_test_accuracy_epoch. With more epochs this architecture easily reaches 97%+ on
MNIST.

The network has #results.trainable_params learnable
parameters — more than enough for MNIST.
