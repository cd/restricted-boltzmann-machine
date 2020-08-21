import RBM from '../RBM.js';

const formatResult = (input) => {
  const states = input.map((e) => e.state);
  if (states[0] && !states[1] && !states[2] && !states[3]) return `one`;
  if (!states[0] && states[1] && !states[2] && !states[3]) return `two`;
  if (!states[0] && !states[1] && states[2] && !states[3]) return `three`;
  if (!states[0] && !states[1] && !states[2] && states[3]) return `four`;
  return 'unknown';
};

// Create RBM
const myRBM = new RBM();

// Train RBM multiple times
for (let index = 0; index < 10000; index++) {
  myRBM.train(
    // Use 4 visible units for 4 numbers
    [
      [1, 0, 0, 0], // "one"
      [0, 1, 0, 0], // "two"
      [0, 0, 1, 0], // "three"
      [0, 0, 0, 1], // "four"
    ],

    // Learning rate
    0.1,

    // Gibbs steps
    1,

    // Use 2 hidden units, because you only need 2 binary digits
    // to represent the information of 4 numbers
    2
  );
}

// Should print every number (in a random order)
console.log(formatResult(myRBM.getVisibleLayer([0, 0])));
console.log(formatResult(myRBM.getVisibleLayer([0, 1])));
console.log(formatResult(myRBM.getVisibleLayer([1, 0])));
console.log(formatResult(myRBM.getVisibleLayer([1, 1])));
