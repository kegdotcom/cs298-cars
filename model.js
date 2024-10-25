// import tf from "@tensorflow/tfjs";
// uncomment above to run the time test in terminal
// comment above to run graphics simulation

export default class PolicyNetwork {

  static actions = ['L', 'R', 'U', 'D'];

  constructor(stateSize = 9, actionSize = PolicyNetwork.actions.length) {
    this.STATE_SIZE = stateSize;
    this.ACTION_SIZE = actionSize;
    this.BATCH_SIZE = 32;
    this.DISCOUNT_FACTOR = tf.scalar(0.1);
    this.LEARNING_RATE = 0.01
    this.network = this.createNetwork();
    this.optimizer = tf.train.adam(this.LEARNING_RATE);
    // this.miniBatches = [];
    this.batch = [];
    this.prevState = tf.variable(tf.tensor2d(new Array(this.STATE_SIZE).fill(NaN), [1, this.STATE_SIZE]), true);
  }

  createNetwork() {
    const model = tf.sequential();

    model.add(tf.layers.dense({
      units: 32,
      inputShape: [this.STATE_SIZE],
      activation: "relu",
    }));

    model.add(tf.layers.dense({
      units: 16,
      activation: "relu",
    }));

    model.add(tf.layers.dense({
      units: this.ACTION_SIZE,
      activation: "softmax",
    }));

    return model;
  }

  // Todo: Justin
  predictActionProbs(state) {
    return this.network.predict(state);
  }

  // predictAction(state) {
  //   const prediction = this.network.predict(state.reshape(1, state.size));
  //   const actionIdx = prediction.argMax(1).dataSync()[0];
  //   return actionIdx;
  // }

  calcReward(state) {
    // calculate the reward from taking action a in state s
    // find vx and vy
    // [d1, d2, d3, d4, d5, x, y, vx, vy];
    const reward = this.prevState.size == 0 ?
      tf.add(state.gather([state.size - 2]).square(), state.gather([state.size - 1]).square()).sqrt()
      :
      tf.add(
        tf.sub(state.gather([state.size - 4]), this.prevState.gather([this.prevState.size - 4])).square(),
        tf.sub(state.gather([state.size - 3]), this.prevState.gather([this.prevState.size - 3])).square()
      ).sqrt();

    this.prevState.assign(state);
    return reward;
  }

  // calcLoss(batch) {
  //   const Q = this.getBatchReward(batch);
  //   const actionProbs = batch.map(tuple => tuple.actionProbs)
  //   const logPr = batch.reduce((acc, tuple) => {
  //     acc += Math.log(tuple.actionProb)
  //   }, 0)
  //   const ElogPr = logPr / batch.length;
  //   const loss = Q * ElogPr;
  //   return -loss;
  // }

  // Q() for minibatch
  getBatchReward(batch) {
    const rewards = tf.concat(batch.map(t => t.reward), 0).reverse();
    const discounts = tf.pow(this.DISCOUNT_FACTOR, tf.range(0, this.BATCH_SIZE));
    const batchRewards = rewards.mul(discounts).cumsum().reverse();
    return batchRewards;
  }

  updatePolicy(batch) {
    this.optimizer.minimize(() => {
      // Extract states and actions from the batch
      const states = tf.stack(batch.map(tuple => tuple.state));
      const actions = tf.concat(batch.map(tuple => tuple.action));
      const actionProbsTensor = tf.stack(batch.map(tuple => tuple.actionProbs));
      const logProbs = actionProbsTensor.mul(tf.oneHot(actions, this.ACTION_SIZE)).sum(-1).log();
      const rewards = this.getBatchReward(batch);

      const loss = rewards.mul(logProbs).mean();
      return tf.neg(loss);
    });
  }

  train(state) {
    // one iteration of the training
    // call once per frame
    // 1. get reward from prev action (a) based on current state (s')
    // 1.2 store previous reward in last entry of batch
    // 2. sample new action and return it (car will take this action)
    // store current (s, a, P(a), r=null) tuple in batch:
    // {state: s, action: x, actionProb: p, reward: y}
    // if that was last action in batch, update last reward and use the batch to update model weights/policy
    // otherwise, continue taking in states (one frame in sim = one state = one call to train())
    // at the end, return the action sampled from current state (s) so that the car can take it
    //  car calls train() here so that each Si gets one train loop, so train gives car back the action

    const stateTensor = tf.tensor2d([state], [1, this.STATE_SIZE]);
    if (this.batch.length > 0) {
      this.batch[this.batch.length - 1].reward = this.calcReward(stateTensor);
    }
    if (this.batch.length === this.BATCH_SIZE) {
      this.updatePolicy(this.batch);
      this.batch = [];
    }

    const actionProbs = this.predictActionProbs(stateTensor);
    const actionIdx = actionProbs.argMax(1);

    this.batch.push({
      state: stateTensor,
      action: actionIdx,
      actionProbs: actionProbs,
      reward: null,
    });
    // sample an action
    return PolicyNetwork.actions[actionIdx.dataSync()[0]];
  }


  /*
  async trainAgent(env, policyNetwork) {
    const batchSize = 32;
    const optimizer = tf.train.adam(0.001);
    
    for (let episode = 0; episode < 5000; episode++) {
      const miniBatch = [];

      for (let i = 0; i < batchSize; i++) {
        miniBatch.push(env.sample());
      }

      optimizer.minimize(() => {
        let totalLoss = 0;
        miniBatch.forEach(([state, action, reward, nextState]) => {
          const [predictedAction, prob] = policyNetwork.predictActionProbs(state);
          reward = this.calcReward(state);
          const loss = this.calcLoss(prob, reward);
          totalLoss += loss;
        });
        return totalLoss / batchSize;
      });
    }
  }
  */
}

// function testTime (iterations) {
//   const start = new Date();
//   const model = new PolicyNetwork();
//   for (let i = 0; i < iterations; i++) {
//     const [action, probs] = model.predictActionProbs([
//       Math.random() * 400,
//       Math.random() * 400,
//       Math.random() * 400,
//       Math.random() * 400,
//       Math.random() * 400,
//       Math.random() * 500,
//       Math.random() * 500,
//       Math.random() * 10,
//       Math.random() * 10,
//     ]);
//     // console.log(`Action ${i+1}:\t${action}\t${(Math.max(...probs) * 100).toFixed(4)}%`);
//   }
//   const end = new Date();
//   console.log(`${iterations} iterations completed in ${(end-start) / 1000} seconds`);
// }
// if (process.argv.length > 2) {
//   testTime(Number(process.argv[2]));
// }
