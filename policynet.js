export default class PolicyNetwork {
  static actions = ['L', 'R', 'U', 'D'];

  constructor(carId, stateSize = 9, actionSize = PolicyNetwork.actions.length) {
    this.STATE_SIZE = stateSize;
    this.ACTION_SIZE = actionSize;
    this.BATCH_SIZE = 32;
    this.DISCOUNT_FACTOR = tf.scalar(0.1, "float32");
    this.LEARNING_RATE = tf.scalar(0.01, "float32");

    this.states = [];

    this.optimizer = tf.train.adam(this.LEARNING_RATE);
    // this.prevState = tf.variable(tf.zeros([1, this.STATE_SIZE]), false, `prevState-${carId}`, "float32");
    this.createPolicyNetwork(this.STATE_SIZE, this.ACTION_SIZE);
  }

  createPolicyNetwork(input, output) {
    this.network = tf.sequential();
    this.network.add(tf.layers.dense({
      units: 32,
      inputShape: [input],
      activation: "relu",
    }));

    this.network.add(tf.layers.dense({
      units: 16,
      activation: "relu",
    }));

    this.network.add(tf.layers.dense({
      units: output,
      activation: "softmax",
    }));
  }

  predict(stateTensor) {
    return tf.tidy(() => {
      const logits = this.network.predict(stateTensor);
      const action = logits.argMax(1);
      return [logits, action];
    });
  }

  calcActionReward(state, statePrime) {
    return tf.tidy(() => {
      const statePosition = tf.slice(state, [5], [2]); // [x, y]
      const statePrimePosition = tf.slice(statePrime, [5], [2]) // [x', y']
      const reward = tf.sub(statePrimePosition, statePosition).square().sum().sqrt();
      return reward;
    });
  }

  discountedRewards(batch) {
    return tf.tidy(() => {
      const positions = batch.slice([0, 5], [-1, 2]);

      const ps = positions.slice([0, 0], [positions.shape[0] - 1, -1]);
      const pPrimes = positions.slice([1, 0]);
      const rewards = tf.sub(pPrimes, ps).square().sum(1).sqrt();
      const n = rewards.shape[0];
      const powers = tf.range(0, n);
      const discounts = tf.pow(this.DISCOUNT_FACTOR, powers);
      for (let i = 0; i < n - 1; i++) {
        const futures = rewards.slice([i + 1]);
        const futureDiscounts = discounts.slice([i + 1]);
        const futureReward = tf.mul(futures, futureDiscounts).sum();
        tf.tensorScatterUpdate(rewards, [i], [futureReward]);
      }
      return rewards;
    });
  }

  updateWeights() {
    this.optimizer.minimize(() => {
      return tf.tidy(() => {
        const statesTensor = tf.concat(this.states, 0);
        const logits = this.network.predict(statesTensor).slice([0], [statesTensor.shape[0] - 1]);
        const actions = logits.argMax(1);
        const logProbs = logits.mul(tf.oneHot(actions, this.ACTION_SIZE)).sum(1).log();
        const rewards = this.discountedRewards(statesTensor);
        const loss = tf.mul(logProbs, rewards).mean();
        return tf.neg(loss);
      });
    })
  }

  runFrame(stateTensor, train = false) {
    if (train) {
      this.states.push(stateTensor);
      if (this.states.length === this.BATCH_SIZE) {
        this.updateWeights();
        this.states = [];
      }
    }
    const [logits, action] = this.predict(stateTensor);
    return action;
  }

  defaultPolicy(state) {
    const [d1, d2, d3, d4, d5, x, y, vx, vy] = state;
    const leftDist = (d1 + d2) / 2;
    const frontDist = (d2 + d3 + d4) / 3;
    const rightDist = (d4 + d5) / 2;
    const speed = Math.hypot(vx, vy);
    let action;
    if (5 * speed > frontDist) {
      // slow down
      action = 'D';
    } else if (10 * speed < frontDist) {
      // speed up
      action = 'U';
    } else {
      if (leftDist > rightDist) {
        // turn left
        action = 'L';
      } else {
        // turn right
        action = 'R';
      }
    }
    return PolicyNetwork.actions.indexOf(action);
  }

  trainOnPolicy(states, policy = this.defaultPolicy) {
    const labels = states.map(policy);
    this.network.fit(states, labels);
  }
}