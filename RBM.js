/**
 * @class
 */
const RBM = (function () {
  'use strict';

  /**
   * Constructor function
   * @param  {array} [weights]
   */
  const RBM = function (weights) {
    this.weights = weights;
  };

  /**
   * RBM version
   */
  RBM.prototype.version = '0.1.0';

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
          arr.push(null);
        } else if (i === 0) {
          arr.push(0);
        } else if (j === 0) {
          arr.push(
            // TODO
            // Math.log(
            //   proportionOfVisibleNodes[i] / (1 - proportionOfVisibleNodes[i])
            // )
            0
          );
        } else {
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
   * Training RBM using CD_1
   * @param {array} dataset Training dataset of visible layers
   * @param {number} [learningRate=0.1] Learning rate
   * @param {number} [hiddenNodes] Number of hidden nodes to init the weights
   * @return {number} Error
   */
  RBM.prototype.train = function (dataset, learningRate = 0.1, hiddenNodes) {
    const errors = [];

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

    dataset.forEach((data) => {
      // Reconstruction
      let hiddenLayer = this.getHiddenLayer(data);
      let visibleLayer = this.getVisibleLayer(hiddenLayer.map((e) => e.state));

      // Add biases
      data = [true, ...data];
      hiddenLayer = [{ state: true, probability: 1 }, ...hiddenLayer];
      visibleLayer = [{ state: true, probability: 1 }, ...visibleLayer];

      // Init error rate
      let error = 0;

      // Loop through every weight (and bias)
      for (let i = 0; i < data.length; i++) {
        for (let j = 0; j < hiddenLayer.length; j++) {
          // Positive statistics
          const positive = data[i] * hiddenLayer[j].probability;

          // Negative statistics
          const negative =
            visibleLayer[i].probability * hiddenLayer[j].probability;

          // Learning rule
          this.weights[i][j] += learningRate * (positive - negative);

          // Update error
          error += Math.pow((data[i] ? 1 : 0) - negative, 2);
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
    const nodes = [true, ...visibleNodes];
    const out = [];
    for (let i = 1; i < this.weights[0].length; i++) {
      let value = 0;
      for (let j = 0; j < nodes.length; j++) {
        if (nodes[j]) value += this.weights[j][i];
      }
      out.push({
        state: this._logistic(value) > Math.random(),
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
    const nodes = [true, ...hiddenNodes];
    const out = [];
    for (let i = 1; i < this.weights.length; i++) {
      let value = 0;
      for (let j = 0; j < nodes.length; j++) {
        if (nodes[j]) value += this.weights[i][j];
      }
      out.push({
        state: this._logistic(value) > Math.random(),
        probability: this._logistic(value),
      });
    }
    return out;
  };

  return RBM;
})();

export default RBM;
