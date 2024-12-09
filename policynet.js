export default class PolicyNetwork {
  static actions = ['L', 'R', 'U', 'D'];

  constructor(carId, stateSize = 9, actionSize = 4 /* PolicyNetwork.actions.length */) {
    this.STATE_SIZE = stateSize;
    this.ACTION_SIZE = actionSize;
    this.BATCH_SIZE = 32;
    this.DISCOUNT_FACTOR = tf.scalar(0.1, "float32");
    this.LEARNING_RATE = tf.scalar(0.01, "float32");
    this.E = 0.1;

    this.states = [];
    this.rewards = [];

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
    this.network.compile({
      optimizer: "sgd",
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"]
    });
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

      const statePrimeCollisions = tf.gather(statePrime, [9])
      const stateCollisions = tf.gather(state, [9]);
      const dCollisions = tf.sub(stateCollisions, statePrimeCollisions);

      const distanceReward = tf.sub(statePrimePosition, statePosition).square().sum().sqrt();
      const collisionPenaltyFactor = dCollisions.add(1);
      const reward = tf.mul(distanceReward, collisionPenaltyFactor);

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

        const avgReward = tf.mean(rewards);
        avgReward.print();

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

  defaultPolicy(state, actionSize = 4) {
    const [d1, d2, d3, d4, d5, x, y, vx, vy, ...data] = state.dataSync();
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
      } else if (rightDist > leftDist) {
        // turn right
        action = 'R';
      } else {
        // speed up
        action = 'U';
      }
    }
    const actionIdx = PolicyNetwork.actions.indexOf(action);
    const actionTensor = tf.oneHot([actionIdx], actionSize);
    return actionTensor;
  }

  async trainOnPolicy(states, log = false, policy = this.defaultPolicy) {
    const labels = states.map(s => policy(s, this.ACTION_SIZE));
    // console.log(states, labels);
    const history = await this.network.fit(states, labels);
    if (log) console.log("History:", history);

    const actions = labels.map(label => label.argMax(1));
    // console.log("pre solo acts", actions);
    return actions;
  }
}

export class GlobalNetwork {
  constructor(nCars, stateSize, actionSize) {
    this.N = nCars;
    this.STATE_SIZE = stateSize;
    this.ACTION_SIZE = actionSize;
    this.BATCH_SIZE = 32;
    this.DISCOUNT_FACTOR = tf.scalar(0.1, "float32");
    this.LEARNING_RATE = tf.scalar(0.01, "float32");

    this.stateStacks = [];
    this.rewards = [];

    this.optimizer = tf.train.adam(this.LEARNING_RATE);
    this.createNetwork();
  }

  createNetwork() {
    const input = tf.input({ shape: [this.N, this.STATE_SIZE] });

    const layer1 = tf.layers.dense({ units: 32, activation: "relu" });
    const layer1Out = layer1.apply(input);

    const layer2 = tf.layers.dense({ units: 16, activation: "relu" });
    const layer2Out = layer2.apply(layer1Out);

    const outputLayer = tf.layers.dense({ units: this.ACTION_SIZE, activation: "softmax" });
    const output = outputLayer.apply(layer2Out);

    this.network = tf.model({ inputs: input, outputs: output });
    this.network.compile({
      optimizer: this.optimizer,
      loss: 'categoricalCrossentropy',
    });
  }

  bulkPredictAction(stateStack, epsilonGreedy = false /* TODO */) {
    const logits = this.network.predict(tf.stack([stateStack])); // shape [1, nCars, actionSize]
    // no epsilon greedy yet, TODO
    const actionsTensor = logits.argMax(2).reshape([this.N]); // shape [nCars]
    return actionsTensor;
  }

  calcActionReward(state, statePrime) {
    return tf.tidy(() => {
      const statePosition = tf.slice(state, [5], [2]); // [x, y]
      const statePrimePosition = tf.slice(statePrime, [5], [2]) // [x', y']

      const statePrimeCollisions = tf.gather(statePrime, [9])
      const stateCollisions = tf.gather(state, [9]);
      const dCollisions = tf.sub(stateCollisions, statePrimeCollisions);

      const distanceReward = tf.sub(statePrimePosition, statePosition).square().sum().sqrt();
      const collisionPenaltyFactor = dCollisions.add(1);
      const reward = tf.mul(distanceReward, collisionPenaltyFactor);

      return reward;
    });
  }

  discountedRewards(stateStackStack) {
    return tf.tidy(() => {
      const positions = stateStackStack.slice([0, 0, 5], [-1, -1, 2]); // shape [batchSize, nCars, 2] (x, y)

      const ps = positions.slice([0, 0, 0], [positions.shape[0] - 1, -1, -1]);
      const pPrimes = positions.slice([1, 0, 0]);
      const distanceRewards = tf.sub(pPrimes, ps).square().sum(2).sqrt(); // shape [batchSize-1, nCars]

      const collisions = stateStackStack.slice([0, 0, this.STATE_SIZE - 1]); // shape [batchSize, nCars, 1]
      const cs = collisions.slice([0, 0, 0], [collisions.shape[0] - 1, -1, -1]); // shape [batchSize-1, nCars, 1]
      const cPrimes = collisions.slice([1, 0, 0]); // shape [batchSize-1, nCars, 1]
      const dCollisions = tf.sub(cPrimes, cs); // shape [batchSize-1, nCars, 1]
      const collisionMultipliers = dCollisions.neg().add(1).sum(2); // shape [batchSize-1, nCars]

      const rewards = tf.mul(distanceRewards, collisionMultipliers); // shape [batchSize-1, nCars]
      const powers = tf.range(0, this.BATCH_SIZE - 1).tile([1, this.N]); // shape [batchSize-1, nCars]
      const discounts = tf.pow(this.DISCOUNT_FACTOR, powers);
      for (let i = 0; i < this.BATCH_SIZE - 1; i++) {
        const futures = rewards.slice([i + 1, 0]);
        const futureDiscounts = discounts.slice([i + 1, 0]);
        const futureRewards = tf.mul(futures, futureDiscounts).sum(0); // shape [nAgents]
        const indices = tf.stack([tf.fill([this.N], i), tf.range(0, this.N)]).transpose(); // shape [nAgents, 2]

        // console.log("fff");
        // rewards.print()
        // indices.print()
        // futureRewards.print()

        tf.tensorScatterUpdate(rewards, indices, futureRewards);
      }
      return rewards;
    });
  }

  updateWeights() {
    this.optimizer.minimize(() => {
      return tf.tidy(() => {
        const stateStackStack = tf.stack(this.stateStacks); // shape [batchSize, nCars, stateSize]
        const logits = this.network.predict(stateStackStack).slice([0, 0, 0], [this.BATCH_SIZE - 1, -1, -1]); // shape [batchSize-1, nCars, actionSize]

        // epsilon-greedy action selection
        const bestActions = logits.argMax(2); // shape [batchSize-1, nCars]
        const randomActions = tf.randomUniformInt([this.BATCH_SIZE - 1, this.N], 0, 4);
        const explore = tf.less(tf.randomUniform([this.BATCH_SIZE - 1, this.N]), this.E);
        const actions = tf.where(explore, randomActions, bestActions); // shape [batchSize-1, nCars]

        const logProbs = logits.mul(tf.oneHot(actions, this.ACTION_SIZE)).sum(2).log(); // shape [batchSize-1, nCars]
        const rewards = this.discountedRewards(stateStackStack); // shape [batchSize-1, nCars]

        // these loss calculations try to optimize the total global reward
        const globalRewards = tf.mul(rewards, logProbs).mean(1); // shape [batchSize-1]
        const avgGlobalReward = tf.mean(globalRewards);
        this.rewards.push(avgGlobalReward);

        // more greedy option (cars care less about the other cars)
        const lossOption1 = tf.mul(rewards, logProbs).mean().neg();
        // mean-subtracted Q function
        const lossOption2 = tf.sub(rewards, tf.mean(rewards)).mul(logProbs).mean().neg();
        return lossOption2;
      });
    })
  }

  runFrame(stateStack, train = false) {
    if (train) {
      this.stateStacks.push(stateStack);
      if (this.stateStacks.length === this.BATCH_SIZE) {
        this.updateWeights();
        console.log("Rewards:", this.rewards);
        this.stateStacks = [];
      }
    }
    const actionsTensor = this.bulkPredictAction(stateStack); // shape [nCars]
    return actionsTensor;
  }

  defaultPolicy(state, actionSize) {
    const [d1, d2, d3, d4, d5, x, y, vx, vy] = state;
    const leftDist = (d1 + d2) / 2;
    const frontDist = (d2 + d3 + d4) / 3;
    const rightDist = (d4 + d5) / 2;
    const speed = Math.hypot(vx, vy);
    let action;
    if (10 * speed > frontDist) {
      // slow down
      action = 'D';
    } else if (20 * speed < frontDist) {
      // speed up
      action = 'U';
    } else {
      if (leftDist > rightDist) {
        // turn left
        action = 'L';
      } else if (rightDist > leftDist) {
        // turn right
        action = 'R';
      } else {
        // speed up
        action = 'U';
      }
    }
    const actionIdx = PolicyNetwork.actions.indexOf(action);
    const actionTensor = tf.oneHot(actionIdx, actionSize);
    return actionTensor;
  }

  async trainOnPolicy(stateStacks, log = false, policy = this.defaultPolicy) {
    const labelTensorLists = stateStacks.map(stack => {
      const states = stack.arraySync();
      return states.map(state => policy(state, this.ACTION_SIZE));
    });
    const labelStacks = labelTensorLists.map(labelTensors => tf.stack(labelTensors));
    const history = await this.network.fit(tf.stack(stateStacks), tf.stack(labelStacks));
    if (log) console.log("Global History:", history);

    const actionsTensors = labelStacks.map(stack => stack.argMax(1));
    // console.log("pre glob acts", actionsTensors);
    return actionsTensors;
  }
}