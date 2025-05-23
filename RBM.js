/**
 * Constructor function
 * @param  {array} [weights] Inital weights
 */
const RBM = function (weights) {
  this.weights = weights;
};

/**
 * RBM version
 */
RBM.prototype.version = 'DEV';

/**
 * Init weights and biases
 * @param {array} proportionOfVisibleNodes
 * @param {number} numberOfHiddenNodes
 */
RBM.prototype._setInitWeights = function (
  proportionOfVisibleNodes,
  numberOfHiddenNodes
) {
  this.weights = [];
  for (let i = 0; i <= proportionOfVisibleNodes.length; i++) {
    const arr = [];
    for (let j = 0; j <= numberOfHiddenNodes; j++) {
      if (i === 0 && j === 0) {
        // No connection between bias units
        arr.push(null);
      } else if (i === 0) {
        // Hidden bias weights
        arr.push(0);
      } else if (j === 0) {
        // Visible bias weights
        arr.push(
          Math.log(
            proportionOfVisibleNodes[i - 1] /
              (1 - proportionOfVisibleNodes[i - 1])
          )
        );
      } else {
        // Regular weights from a zero-mean Gaussian with
        // standard deviation of 0.01
        arr.push(this._gaussianRand(0.01));
      }
    }
    this.weights.push(arr);
  }
};

/**
 * Logistic function
 * @param {number} x
 * @return {number}
 */
RBM.prototype._logistic = function (x) {
  return 1 / (1 + Math.exp(-x));
};

/**
 * Random number from an approximated gaussian standard normal distribution
 * @param {number} standardDeviation
 * @return {number}
 */
RBM.prototype._gaussianRand = function (standardDeviation) {
  let rand = 0;
  for (let i = 0; i < 6; i += 1) {
    rand += Math.random();
  }
  return -standardDeviation + (rand / 6) * standardDeviation * 2;
};

/**
 * Training RBM using CD_n
 * @param {array} dataset Training dataset of visible layers
 * @param {number} learningRate Learning rate
 * @param {number} gibbsSteps Full steps of Gibbs sampling
 * @param {number} [hiddenNodes] Number of hidden nodes to init the weights
 * @return {number} Error rate
 */
RBM.prototype.train = function (
  dataset,
  learningRate,
  gibbsSteps,
  hiddenNodes
) {
  const errors = [];

  // Init weights if RBM isn't initialized
  if (!this.weights && hiddenNodes) {
    const proportions = new Array(dataset[0].length).fill(0);
    for (let i = 0; i < proportions.length; i++) {
      for (let j = 0; j < dataset.length; j++) {
        proportions[i] += dataset[j][i];
      }
      proportions[i] /= dataset.length;
    }
    this._setInitWeights(proportions, hiddenNodes);
  }

  // Loop through every training dataset
  dataset.forEach((data) => {
    // Postive CD phase
    let visibleLayerPos = data;
    let hiddenLayerPos = this.getHiddenLayer(data);

    // Negative CD phase
    let visibleLayerNeg = data;
    let hiddenLayerNeg;
    for (let i = 0; i < gibbsSteps; i++) {
      hiddenLayerNeg = this.getHiddenLayer(
        visibleLayerNeg.map((e) => e.probability || e)
      );
      visibleLayerNeg = this.getVisibleLayer(
        hiddenLayerNeg.map((e) => e.state)
      );
    }

    // Add biases
    visibleLayerPos = [1, ...visibleLayerPos];
    hiddenLayerPos = [{ state: 1, probability: 1 }, ...hiddenLayerPos];
    visibleLayerNeg = [{ state: 1, probability: 1 }, ...visibleLayerNeg];
    hiddenLayerNeg = [{ state: 1, probability: 1 }, ...hiddenLayerNeg];

    // Init error rate
    let error = 0;

    // Loop through every weight (and bias)
    for (let i = 0; i < visibleLayerPos.length; i++) {
      // Do not update weights if visible node is negative
      if (visibleLayerPos[i] < 0) continue;

      for (let j = 0; j < hiddenLayerPos.length; j++) {
        // Positive statistic
        const positive = visibleLayerPos[i] * hiddenLayerPos[j].probability;

        // Negative statistic
        const negative =
          visibleLayerNeg[i].probability * hiddenLayerNeg[j].probability;

        // Learning rule
        this.weights[i][j] += learningRate * (positive - negative);

        // Update error rate
        error += Math.pow((visibleLayerPos[i] ? 1 : 0) - negative, 2);
      }
    }

    errors.push(error);
  });

  return errors;
};

/**
 * Update the hidden nodes using the visible binary units
 * @param {array} visibleNodes
 * @return {array} Hidden nodes as objects of {state, probability}
 */
RBM.prototype.getHiddenLayer = function (visibleNodes) {
  if (
    !Array.isArray(visibleNodes) ||
    visibleNodes.length !== this.weights.length - 1 ||
    visibleNodes.length === 0
  ) {
    throw new Error('Invalid parameter');
  }

  // Add bias unit
  const nodes = [1, ...visibleNodes];

  // Init array of hidden nodes
  const out = [];

  // Loop through every hidden unit
  for (let i = 1; i < this.weights[0].length; i++) {
    // Init activisition energy
    let value = 0;

    for (let j = 0; j < nodes.length; j++) {
      // Ignore negative visible nodes
      if (nodes[j] < 0) continue;

      // Update activisition energy
      value += this.weights[j][i] * nodes[j];
    }

    // Calc hidden unit
    out.push({
      state: this._logistic(value) > Math.random() ? 1 : 0,
      probability: this._logistic(value),
    });
  }
  return out;
};

/**
 * Update the visible nodes using the hidden binary units
 * @param {array} hiddenNodes
 * @return {array} Visible nodes as objects of {state, probability}
 */
RBM.prototype.getVisibleLayer = function (hiddenNodes) {
  if (
    !Array.isArray(hiddenNodes) ||
    hiddenNodes.length !== this.weights[0].length - 1 ||
    hiddenNodes.length === 0
  ) {
    throw new Error('Invalid parameter');
  }

  // Add bias unit
  const nodes = [1, ...hiddenNodes];

  // Init array of visible nodes
  const out = [];

  // Loop through every visible unit
  for (let i = 1; i < this.weights.length; i++) {
    // Init activisition energy
    let value = 0;

    // Add weights
    for (let j = 0; j < nodes.length; j++) {
      value += this.weights[i][j] * nodes[j];
    }

    // Calc visible unit
    out.push({
      state: this._logistic(value) > Math.random() ? 1 : 0,
      probability: this._logistic(value),
    });
  }
  return out;
};

export default RBM;
