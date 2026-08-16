// Workgroup score arrays for the krystal_pool_backward shader. Record width is
// the frozen ABI value 8, so each independent 8-slot array fits the key and
// value pooling rows. Module-scope because Sandblaster injects shader bodies
// inside the entry function.
//
//   poolKeyScores/poolValueScores   softmax probabilities (recomputed, same
//                                   math as krystal_pool forward)
//   poolKeyGrads/poolValueGrads     dScoreK/dScoreV after the softmax
//                                   backward, from the recomputed probs
var<workgroup> poolKeyScores: array<f32, 8>;
var<workgroup> poolValueScores: array<f32, 8>;
var<workgroup> poolKeyGrads: array<f32, 8>;
var<workgroup> poolValueGrads: array<f32, 8>;
