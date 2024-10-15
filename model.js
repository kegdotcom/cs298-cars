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
  
  predictActionProb (carData) {
    const inputData = tf.tensor2d([carData]);
    const prediction = this.network.predict(inputData);
    const actionIdx = prediction.argMax(1).dataSync()[0];
    const action = PolicyNetwork.actions[actionIdx];
    const actionProb = prediction.max().dataSync()[0];
    return [action, actionProb];
  }
  
  predictActionProbs (carData) {
    const inputData = tf.tensor2d([carData]);
    const prediction = this.network.predict(inputData);
    const probs = prediction.dataSync();
    const actionIdx = prediction.argMax(1).dataSync()[0];
    const action = PolicyNetwork.actions[actionIdx];
    return [action, probs];
  }
  
  calcReward (s, a) {
    // calculate the reward from taking action a in state s
    // find vx and vy
    const velocity_xy = [s[s.length - 2],s[s.length - 1]];
    const velocity = Math.sqrt(velocity_xy[0]**2 + velocity_xy[1]**2);
    return velocity;
  }

  calcLoss (prob, reward) {
    // return the NEGATIVE log times the reward. 
    return -math.log(prob) * reward;
  }

  async updatePolicy (s, a, r, sPrime) {
    // use policy and rewards and loss to update network weights in training
    // SGD for pretraining. 
    const optimizer = tf.train.sgd(0.01)

    optimizer.minimize(() => {
      const [action,prob] = this.predictActionProb(s);
      const reward = this.calcReward(s,action);
      // now calculate the loss
      const loss = this.calcLoss(prob,reward);
      return loss;
    })

  }

  async trainAgent(env, policyNetwork){
    // TODO: training loop
  }
}

function testTime (iterations) {
  const start = new Date();
  const model = new PolicyNetwork();
  for (let i = 0; i < iterations; i++) {
    const [action, probs] = model.predictActionProbs([
      Math.random() * 400,
      Math.random() * 400,
      Math.random() * 400,
      Math.random() * 400,
      Math.random() * 400,
      Math.random() * 500,
      Math.random() * 500,
      Math.random() * 10,
      Math.random() * 10,
    ]);
    // console.log(`Action ${i+1}:\t${action}\t${(Math.max(...probs) * 100).toFixed(4)}%`);
  }
  const end = new Date();
  console.log(`${iterations} iterations completed in ${(end-start) / 1000} seconds`);
}
// if (process.argv.length > 2) {
//   testTime(Number(process.argv[2]));
// }