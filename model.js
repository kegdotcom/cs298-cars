
class PolicyNetwork {
  constructor (stateSize = 9, actionSize = 4) {
    this.stateSize = stateSize;
    this.actionSize = actionSize;
    this.model = this.createModel(this.stateSize, this.actionSize);
  }

  createModel (stateSize, actionSize) {
    const model = tf.sequential();
      
    model.add(tf.layers.dense({
      units: 24,
      inputShape: [stateSize],
      activation: "relu",
    }));
      
    model.add(tf.layers.dense({
      units: 24,
      activation: "relu",
    }));
      
    model.add(tf.layers.dense({
      units: actionSize,
      activation: "softmax",
    }));
  
    return model;
  }
  // Todo: keg
  static calcLoss () {
    
  }
  // Todo: Justin
  async predictAction (state, policyNetwork) {

  }

  async updatePolicy (s, a, r, sPrime) {
    
  }

  async trainAgent(env, policyNetwork){
    
  }
}

