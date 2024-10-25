import tf from "@tensorflow/tfjs";
// uncomment above to run the time test in terminal
// comment above to run graphics simulation

export default class PolicyNetwork {
  static batchSize = 32;
  static discountFactor = 0.1;
  static learningRate = 0.01

  static actions = ['L', 'R', 'U', 'D'];

  constructor(stateSize = 9, actionSize = PolicyNetwork.actions.length) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.network = this.createNetwork(this.stateSize, this.actionSize);
    this.optimizer = tf.train.adam(PolicyNetwork.learningRate);
    this.miniBatches = [];
    this.batch = [];
    this.prevPos = null;
  }

  createNetwork(stateSize, actionSize) {
    const model = tf.sequential();

    model.add(tf.layers.dense({
      units: 32,
      inputShape: [stateSize],
      activation: "relu",
    }));

    model.add(tf.layers.dense({
      units: 16,
      activation: "relu",
    }));

    model.add(tf.layers.dense({
      units: actionSize,
      activation: "softmax",
    }));

    return model;
  }

  // Todo: Justin
  predictProbs(carData) {
    const inputData = tf.tensor2d([carData], [1, 9]);
    const prediction = this.network.predict(inputData)
    return prediction.dataSync();
  }

  predictAction(carData) {
    const inputData = tf.tensor2d([carData], [1, 9]);
    const prediction = this.network.predict(inputData);
    const actionIdx = prediction.argMax(1).dataSync()[0];
    return PolicyNetwork.actions[actionIdx];
  }

  calcReward(s) {
    // calculate the reward from taking action a in state s
    // find vx and vy
    // [d1, d2, d3, d4, d5, x, y, vx, vy];
    let reward;
    const newPos = [s[s.length - 4], s[s.length - 3]];
    if (this.prevPos === null) {
      reward = Math.hypot(s[s.length - 2], s[s.length - 1]);
    } else {
      reward = Math.hypot(newPos[1] - this.prevPos[1], newPos[0] - this.prevPos[0]);
    }
    this.prevPos = newPos;
    return reward;
  }

  calcLoss(batch) {
    const Q = this.getBatchReward(batch);
    const logPr = batch.reduce((acc, tuple) => {
      acc += Math.log(tuple.actionProb)
    }, 0)
    const ElogPr = logPr / batch.length;
    const loss = Q * ElogPr;
    return -loss;
  }

  predictActionProb(carData) {
    const inputData = tf.tensor2d([carData], [1, 9]);
    const prediction = this.network.predict(inputData);
    const probs = prediction.dataSync();
    const actionIdx = prediction.argMax(1).dataSync()[0];
    return [PolicyNetwork.actions[actionIdx], probs[actionIdx]];
  }

  // Q() for minibatch
  getBatchReward(batch) {
    let q = 0;
    batch.forEach((tuple, idx) => {
      q += (PolicyNetwork.discountFactor ** idx) * tuple.reward;
    })
    return q;
  }

  async updatePolicy(batch) {
    optimizer.minimize(() => {
      const loss = this.calcLoss(batch);
      return loss;
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

    if (this.batch.length > 0) {
      this.batch[this.batch.length - 1].reward = this.calcReward(state);
    }
    if (this.batch.length == PolicyNetwork.batchSize) {
      this.updatePolicy(this.batch);
      this.batch = [];
    }

    const [action, p] = this.predictActionProb(state);
    this.batch.push({
      state: state,
      action: action,
      actionProb: p,
      reward: null,
    });
    // sample an action
    return action;
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
