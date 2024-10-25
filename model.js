// import tf from "@tensorflow/tfjs";
// uncomment above to run the time test in terminal
// comment above to run graphics simulation

export default class PolicyNetwork {
  static actions = ['L', 'R', 'U', 'D'];
  constructor (stateSize = 9, actionSize = PolicyNetwork.actions.length) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.network = this.createNetwork(this.stateSize, this.actionSize);
  }

  createNetwork (stateSize, actionSize) {
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
  predictProbs (carData) {
    const inputData = tf.tensor2d([carData]);
    const prediction = this.network.predict(inputData)
    return prediction.dataSync();
  }
  
  predictAction (carData) {
    const inputData = tf.tensor2d([carData]);
    const prediction = this.network.predict(inputData);
    const actionIdx = prediction.argMax(1).dataSync()[0];
    return PolicyNetwork.actions[actionIdx];
  }

  calcReward(s) {
    // calculate the reward from taking action a in state s
    // find vx and vy
    const velocity_xy = Math.sqrt(s[s.length - 2]**2 + s[s.length - 1]**2);
    return velocity_xy;
  }

  calcLoss (prob, reward) {
    // return the NEGATIVE log times the reward. 
    return tf.neg(tf.mul(tf.log(prob), reward));
  }

  predictActionProbs (carData) {
    const inputData = tf.tensor2d([carData]);
    const prediction = this.network.predict(inputData);
    const probs = prediction.dataSync();
    const actionIdx = prediction.argMax(1).dataSync()[0];
    const action = PolicyNetwork.actions[actionIdx];
    return [action, probs];
  }
  

  async updatePolicy(state, action, reward) {
    const optimizer = tf.train.adam(0.001);  // Adam optimizer for policy gradient

    optimizer.minimize(() => {
      const [predictedAction, prob] = this.predictActionProbs(state);
      const calculatedReward = this.calcReward(state);
      const loss = this.calcLoss(prob, calculatedReward);
      return loss;
    });
  }

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
}