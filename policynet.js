// import fs from "fs";

export default class PolicyNetwork {
  static actions = ['L', 'R', 'U', 'D'];

  constructor(carId, stateSize = 10, actionSize = 4 /* PolicyNetwork.actions.length */) {
    this.STATE_SIZE = stateSize;
    this.ACTION_SIZE = actionSize;
    this.BATCH_SIZE = 128;
    this.DISCOUNT_FACTOR = 0.1; //tf.scalar(0.1, "float32");
    this.LEARNING_RATE = 0.01; //tf.scalar(0.01, "float32");
    this.E = 0.1;

    this.states = [];
    this.reward;

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
      if (this.states.length >= this.BATCH_SIZE) {
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
  static actions = ['L', 'R', 'U', 'D']
  constructor(load, nCars, stateSize, actionSize) {
    this.N = nCars;
    this.STATE_SIZE = stateSize;
    this.ACTION_SIZE = actionSize;
    this.BATCH_SIZE = 32;
    this.DISCOUNT_FACTOR = 0.9;
    this.LEARNING_RATE = 0.001;
    this.E = 0.1

    this.stateStacks = [];
    this.labelStacks = [];
    this.avgRewards = [0, 0, 0];
    this.nRewards = [0, 0, 0];

    this.optimizer = tf.train.adam(this.LEARNING_RATE);
    if (!load) {
      this.createNetwork();
    }
    this.actions = 0;
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

  bulkPredictAction(stateStack, epsilonGreedy = false) {
    return tf.tidy(() => {
      const x = tf.stack([stateStack]); // shape [1, nCars, stateSize]
      // x.print();
      const logits = this.network.predict(x); // shape [1, nCars, actionSize]
      const actions = tf.tidy(() => {
        if (epsilonGreedy) {
          const bestActions = logits.argMax(2).reshape([this.N]);
          const randomActions = tf.randomUniformInt([this.N], 0, 4);
          const explore = tf.less(tf.randomUniform([this.N]), this.E);
          return tf.where(explore, randomActions, bestActions);
        } else {
          return logits.argMax(2).reshape([this.N]);
        }
      });
      // logits.print();
      return actions;
    });
  }

  calcRewards(stateStackStack, speedFactor = 50) {
    return tf.tidy(() => {
      const positions = stateStackStack.slice([0, 0, 5], [-1, -1, 2]); // shape [batchSize, nCars, 2] (x, y)

      const ps = positions.slice([0, 0, 0], [positions.shape[0] - 1, -1, -1]);
      const pPrimes = positions.slice([1, 0, 0]);
      const distanceRewards = tf.sub(pPrimes, ps).square().sum(2).sqrt().mul(speedFactor); // shape [batchSize-1, nCars]

      const startCollisions = stateStackStack.slice([0, 0, this.STATE_SIZE - 1], [1, -1, 1]).reshape([this.N]);
      const endCollisions = stateStackStack.slice([this.BATCH_SIZE - 1, 0, this.STATE_SIZE - 1], [1, -1, 1]).reshape([this.N]);
      const collisionFactor = tf.sub(endCollisions, startCollisions).add(1).pow(2);
      const rewards = tf.div(distanceRewards, collisionFactor);
      const avgRewards = rewards.mean(0).arraySync();
      this.avgRewards = this.avgRewards.map((rew, idx) => {
        return rew + (avgRewards[idx] / ++this.nRewards[idx]);
      });
      return rewards;
    });
  }

  discountRewards(stateStackStack) {
    return tf.tidy(() => {
      const rewards = this.calcRewards(stateStackStack);
      const powers = tf.range(0, this.BATCH_SIZE - 1).reshape([this.BATCH_SIZE - 1, 1]).tile([1, this.N]); // shape [batchSize-1, nCars] each col is 0-30
      const discounts = tf.pow(this.DISCOUNT_FACTOR, powers);
      for (let i = 0; i < this.BATCH_SIZE - 2; i++) {
        const futures = rewards.slice([i, 0]);
        const futureDiscounts = discounts.slice([i, 0]);
        const futureRewards = tf.mul(futures, futureDiscounts).sum(0) // shape [nAgents]
        const indices = tf.stack([tf.fill([this.N], i, "int32"), tf.range(0, this.N, 1, "int32")]).transpose(); // shape [nAgents, 2]
        tf.tensorScatterUpdate(rewards, indices, futureRewards);
      }
      return rewards;
    });
  }

  updateWeights(stateStacks) {
    console.log("MINIMIZING")
    this.optimizer.minimize(() => {
      return tf.tidy(() => {
        const stateStackStack = tf.stack(stateStacks); // shape [batchSize, nCars, stateSize]
        const logits = this.network.predict(stateStackStack).slice([0, 0, 0], [this.BATCH_SIZE - 1, -1, -1]); // shape [batchSize-1, nCars, actionSize]

        // epsilon-greedy action selection
        const bestActions = logits.argMax(2); // shape [batchSize-1, nCars]
        const randomActions = tf.randomUniformInt([this.BATCH_SIZE - 1, this.N], 0, 4);
        const explore = tf.less(tf.randomUniform([this.BATCH_SIZE - 1, this.N]), this.E);
        const actions = tf.where(explore, randomActions, bestActions); // shape [batchSize-1, nCars]

        const oneHotActions = tf.oneHot(bestActions, this.ACTION_SIZE);
        const probs = logits.clipByValue(0.1, 0.9).mul(oneHotActions)
        const logProbs = probs.sum(2).log(); // shape [batchSize-1, nCars]
        // console.log('probs', logits.arraySync(), oneHotActions.arraySync(), probs.arraySync(), logProbs.arraySync())
        // const rewards = this.calcRewards(stateStackStack);
        const rewards = this.discountRewards(stateStackStack);

        // console.log("steps", rewards.arraySync(), logProbs.arraySync())
        // more greedy option (cars care less about the other cars)
        const lossOption1 = tf.mul(rewards, logProbs).mean(0).sum().neg();

        // mean-subtracted Q function
        const lossOption2 = tf.sub(rewards, tf.mean(rewards)).mul(logProbs).mean().neg();
        // lossOption2.print()
        console.log("Loss");
        lossOption1.print()
        return lossOption1;
      });
    })
  }

  async runFrame(stateStack, train = true) {
    if (this.network === undefined) {
      await this.loadModel();
    }
    if (++this.actions % 2500 === 0) {
      await this.network.save("localstorage://model");
      this.printWeights();
    };
    // this.printWeights();
    if (train) {
      this.stateStacks.push(stateStack);
      // console.log(this.stateStacks.length, this.BATCH_SIZE);
      if (this.stateStacks.length === this.BATCH_SIZE) {
        this.updateWeights(this.stateStacks);
        // this.stateStacks.forEach(stack => {
        //   stack.dispose();
        // });
        this.stateStacks.length = 0;
      }
    }
    const actionsTensor = this.bulkPredictAction(stateStack, true); // shape [nCars]
    // actionsTensor.print();
    return actionsTensor;
  }

  defaultPolicy(state) {
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
        action = 'D';
      }
    }
    return GlobalNetwork.actions.indexOf(action);
  }

  async trainOnPolicy(stateStacks, record = false, policy = this.defaultPolicy) {
    if (++this.actions % 2500 === 0) this.printWeights();
    const [x, y, actions] = tf.tidy(() => {
      // if (record) {
      // const writeStream = fs.createWriteStream("pretraining-states.json", { flags: 'a' });
      // stateStacks.forEach(stack => {
      // const stateList = stack.arraySync();
      // writeStream.write(JSON.stringify(stateList) + '\n');
      // })
      // writeStream.end();
      // }

      this.stateStacks.push(...stateStacks);
      const actionLists = stateStacks.map(stateStack => {
        const states = stateStack.arraySync();
        const actionIdxs = states.map(policy);
        return actionIdxs;
      });

      // if (record) {
      //   const writeStream = fs.createWriteStream("pretraining-labels.json", { flags: 'a' });
      //   actionLists.forEach(actions => {
      //     writeStream.write(JSON.stringify(actions) + '\n');
      //   })
      //   writeStream.end();
      // }

      const labelStacks = actionLists.map(actions => tf.oneHot(actions, this.ACTION_SIZE));
      this.labelStacks.push(...labelStacks);

      const stateMatrix = tf.stack(stateStacks);
      const labelMatrix = tf.stack(labelStacks);
      return [stateMatrix, labelMatrix, actionLists];
    });

    if (this.stateStacks.length === this.BATCH_SIZE) {
      await this.network.fit(x, y, {
        batchSize: 32,
        epochs: 3
      });
      x.dispose();
      y.dispose();

      // this.stateStacks.forEach(stack => {
      //   stack.dispose();
      // });
      this.stateStacks.length = 0;
      // this.labelStacks.forEach(stack => {
      //   stack.dispose();
      // });
      this.labelStacks.length = 0;
    }

    // console.log("pre glob acts", actionsTensors);
    return actions;
  }

  async loadModel() {
    this.network = await tf.loadLayersModel("localstorage://model");
  }

  getRewards() {
    return this.rewards.map(arraySync);
  }
  printWeights() {
    const mem = tf.memory();
    console.log("memory", mem.numTensors, mem);
    this.network.getWeights().forEach((weight, layer) => {
      console.log(`Layer ${layer} weights: ${weight.toString()}`);
    });
    console.log("rew", this.avgRewards);
  }
}