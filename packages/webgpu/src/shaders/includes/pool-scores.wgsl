// Workgroup score arrays for the krystal_pool shader. Record width is the
// frozen ABI value 8, so two independent arrays of 8 fit the key and value
// pooling score rows. Module-scope because Sandblaster injects shader bodies
// inside the entry function.
var<workgroup> poolKeyScores: array<f32, 8>;
var<workgroup> poolValueScores: array<f32, 8>;
