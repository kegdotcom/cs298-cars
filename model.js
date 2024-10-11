import * as tf from "@tensorflow/tfjs";

const model = tf.sequential();

model.add(tf.layers.dense({
  units: 36,
  inputShape: [9],
  activation: "relu",
}));

model.add(tf.layers.dense({
  units: 12,
  activation: "relu",
}));

model.add(tf.layers.dense({
  units: 4,
  activation: "softmax",
}));

model.compile({
  optimizer: 'adam',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

export default model;