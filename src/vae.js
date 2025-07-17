// VAE in tensorflow.js
// based on https://github.com/songer1993/tfjs-vae

const Max = require("max-api");
const tf = require("@tensorflow/tfjs-node");

const utils = require("./utils.js");
const data = require("./data.js");

// Constants
const NUM_MIDI_CLASSES = require("./constants.js").NUM_MIDI_CLASSES;
const LOOP_DURATION = require("./constants.js").LOOP_DURATION;

const ORIGINAL_DIM = require("./constants.js").ORIGINAL_DIM;

// Constants - Optimized for M1 Max performance
const BATCH_SIZE = 128; // Increased from 64 for better M1 Max utilization
const TEST_BATCH_SIZE = 256; // Increased from 128
const INTERMEDIATE_DIM = 1024; // Increased from 512 for better M1 Max performance
const LATENT_DIM = 2;

// Constants - Loss coefficients
const ON_LOSS_COEF = 1.0; // Standard weight for onset
const VEL_LOSS_COEF = 2.5; // Standard weight for velocity
const DUR_LOSS_COEF = 2.5; // Standard weight for duration
const TIME_LOSS_COEF = 5.0; // Same as RhythmVAE timeshift coefficient

let dataHandlerOnset;
let dataHandlerVelocity;
let dataHandlerDuration;
let dataHandlerTimeshift;
let model = null;
let numEpochs = 150; // Match RhythmVAE default

async function loadAndTrain(
  train_data_onset,
  train_data_velocity,
  train_data_duration,
  train_data_timeshift
) {
  console.assert(
    train_data_onset.length == train_data_velocity.length &&
      train_data_velocity.length == train_data_duration.length &&
      train_data_duration.length == train_data_timeshift.length
  );

  // shuffle in sync
  const total_num = train_data_onset.length;
  shuffled_indices = tf.util.createShuffledIndices(total_num);
  train_data_onset = utils.shuffle_with_indices(
    train_data_onset,
    shuffled_indices
  );
  train_data_velocity = utils.shuffle_with_indices(
    train_data_velocity,
    shuffled_indices
  );
  train_data_duration = utils.shuffle_with_indices(
    train_data_duration,
    shuffled_indices
  );
  train_data_timeshift = utils.shuffle_with_indices(
    train_data_timeshift,
    shuffled_indices
  );

  // synced indices
  const num_trains = Math.floor(data.TRAIN_TEST_RATIO * total_num);
  const num_tests = total_num - num_trains;
  const train_indices = tf.util.createShuffledIndices(num_trains);
  const test_indices = tf.util.createShuffledIndices(num_tests);

  // create data handlers
  dataHandlerOnset = new data.DataHandler(
    train_data_onset,
    train_indices,
    test_indices
  );
  dataHandlerVelocity = new data.DataHandler(
    train_data_velocity,
    train_indices,
    test_indices
  );
  dataHandlerDuration = new data.DataHandler(
    train_data_duration,
    train_indices,
    test_indices
  );
  dataHandlerTimeshift = new data.DataHandler(
    train_data_timeshift,
    train_indices,
    test_indices
  );

  // start training!
  if (!model) initModel(); // initializing model class
  startTraining(); // start the actual training process with the given training data
}

function initModel() {
  model = new ConditionalVAE({
    modelConfig: {
      originalDim: ORIGINAL_DIM,
      intermediateDim: INTERMEDIATE_DIM,
      latentDim: LATENT_DIM,
    },
    trainConfig: {
      batchSize: BATCH_SIZE,
      testBatchSize: TEST_BATCH_SIZE,
      optimizer: tf.train.adam(),
    },
  });
}

async function startTraining() {
  await model.train();
}

function stopTraining() {
  model.shouldStopTraining = true;
  utils.log_status("Stopping training...");
}

function isTraining() {
  if (model && model.isTraining) return true;
}

function isReadyToGenerate() {
  return model && model.isTrained;
}

function setEpochs(e) {
  numEpochs = e;
  Max.outlet("epoch", 0, numEpochs);
}

function generatePattern(z1, z2, noise_range = 0.0) {
  var zs;
  if (z1 === "undefined" || z2 === "undefined") {
    zs = tf.randomNormal([1, 2]);
  } else {
    zs = tf.tensor2d([[z1, z2]]);
  }

  // noise
  if (noise_range > 0.0) {
    var noise = tf.randomNormal([1, 2]);
    zs = zs.add(noise.mul(tf.scalar(noise_range)));
  }
  return model.generate(zs);
}

function encodePattern(inputOn, inputVel, inputDur, inputTime) {
  return model.encode(inputOn, inputVel, inputDur, inputTime);
}

function clearModel() {
  model = null;
}

function bendModel(noise_range) {
  model.bendModel(noise_range);
}

async function saveModel(filepath) {
  model.saveModel(filepath);
}

async function loadModel(filepath) {
  if (!model) initModel();
  model.loadModel(filepath);
}

// Sampling Z
class sampleLayer extends tf.layers.Layer {
  constructor(args) {
    super({});
  }

  computeOutputShape(inputShape) {
    return inputShape[0];
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      const [zMean, zLogVar] = inputs;
      const batch = zMean.shape[0];
      const dim = zMean.shape[1];
      const epsilon = tf.randomNormal([batch, dim]);
      const half = tf.scalar(0.5);
      const temp = zLogVar.mul(half).exp().mul(epsilon);
      const sample = zMean.add(temp);
      return sample;
    });
  }

  getClassName() {
    return "sampleLayer";
  }
}

class ConditionalVAE {
  constructor(config) {
    this.modelConfig = config.modelConfig;
    this.trainConfig = config.trainConfig;
    [this.encoder, this.decoder, this.apply] = this.build();
    this.isTrained = false;
  }

  build(modelConfig) {
    if (modelConfig != undefined) {
      this.modelConfig = modelConfig;
    }
    const config = this.modelConfig;

    const originalDim = config.originalDim;
    const intermediateDim = config.intermediateDim;
    const latentDim = config.latentDim;

    // VAE model = encoder + decoder
    // build encoder model - SIMPLIFIED like RhythmVAE

    const createEncoderBranch = (name) => {
      const input = tf.input({
        shape: [originalDim],
        name: `encoder_input_${name}`,
      });
      const x1Linear = tf.layers
        .dense({
          units: intermediateDim,
          useBias: true,
          kernelInitializer: "glorotNormal",
        })
        .apply(input);
      const x1Normalised = tf.layers
        .batchNormalization({ axis: 1 })
        .apply(x1Linear);
      const x1 = tf.layers.leakyReLU().apply(x1Normalised);
      return [input, x1];
    };

    const [encoderInputsOn, x1On] = createEncoderBranch("on");
    const [encoderInputsVel, x1Vel] = createEncoderBranch("vel");
    const [encoderInputsDur, x1Dur] = createEncoderBranch("dur");
    const [encoderInputsTime, x1Time] = createEncoderBranch("time");

    // Merged - concatenate all 4 inputs
    const concatLayer = tf.layers.concatenate();
    const x1Merged = concatLayer.apply([x1On, x1Vel, x1Dur, x1Time]);
    const x2Linear = tf.layers
      .dense({
        units: intermediateDim,
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(x1Merged);
    const x2Normalised = tf.layers
      .batchNormalization({ axis: 1 })
      .apply(x2Linear);
    const x2 = tf.layers.leakyReLU().apply(x2Normalised);

    const zMean = tf.layers
      .dense({
        units: latentDim,
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(x2);
    const zLogVar = tf.layers
      .dense({
        units: latentDim,
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(x2);
    const z = new sampleLayer().apply([zMean, zLogVar]);

    const encoderInputs = [
      encoderInputsOn,
      encoderInputsVel,
      encoderInputsDur,
      encoderInputsTime,
    ];
    const encoderOutputs = [zMean, zLogVar, z];

    const encoder = tf.model({
      inputs: encoderInputs,
      outputs: encoderOutputs,
      name: "encoder",
    });

    // build decoder model - SIMPLIFIED like RhythmVAE
    const decoderInputs = tf.input({ shape: [latentDim] });
    const x3Linear = tf.layers
      .dense({
        units: intermediateDim * 2.0,
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(decoderInputs);
    const x3Normalised = tf.layers
      .batchNormalization({ axis: 1 })
      .apply(x3Linear);
    const x3 = tf.layers.leakyReLU().apply(x3Normalised);

    const createDecoderBranch = (input, activation, name) => {
      const x4Linear = tf.layers
        .dense({
          units: intermediateDim,
          useBias: true,
          kernelInitializer: "glorotNormal",
        })
        .apply(input);
      const x4Normalised = tf.layers
        .batchNormalization({ axis: 1 })
        .apply(x4Linear);
      const x4 = tf.layers.leakyReLU().apply(x4Normalised);
      return tf.layers
        .dense({
          units: originalDim,
          activation,
          name: `decoder_output_${name}`,
        })
        .apply(x4);
    };

    const decoderOutputsOn = createDecoderBranch(x3, "sigmoid", "on");
    const decoderOutputsVel = createDecoderBranch(x3, "sigmoid", "vel");
    const decoderOutputsDur = createDecoderBranch(x3, "sigmoid", "dur");
    const decoderOutputsTime = createDecoderBranch(x3, "tanh", "time");

    const decoderOutputs = [
      decoderOutputsOn,
      decoderOutputsVel,
      decoderOutputsDur,
      decoderOutputsTime,
    ];

    // Decoder model
    const decoder = tf.model({
      inputs: decoderInputs,
      outputs: decoderOutputs,
      name: "decoder",
    });

    // build VAE model
    const vae = (inputs) => {
      return tf.tidy(() => {
        const [zMean, zLogVar, z] = this.encoder.apply(inputs);
        const outputs = this.decoder.apply(z);
        return [zMean, zLogVar, outputs];
      });
    };

    return [encoder, decoder, vae];
  }

  reconstructionLoss(yTrue, yPred) {
    return tf.tidy(() => {
      let reconstruction_loss;
      reconstruction_loss = tf.metrics.binaryCrossentropy(yTrue, yPred);
      reconstruction_loss = reconstruction_loss.mul(tf.scalar(yPred.shape[1]));
      return reconstruction_loss;
    });
  }

  mseLoss(yTrue, yPred) {
    return tf.tidy(() => {
      let mse_loss = tf.metrics.meanSquaredError(yTrue, yPred);
      mse_loss = mse_loss.mul(tf.scalar(yPred.shape[1]));
      return mse_loss;
    });
  }

  klLoss(z_mean, z_log_var) {
    return tf.tidy(() => {
      let kl_loss;
      kl_loss = tf
        .scalar(1)
        .add(z_log_var)
        .sub(z_mean.square())
        .sub(z_log_var.exp());
      kl_loss = tf.sum(kl_loss, -1);
      kl_loss = kl_loss.mul(tf.scalar(-0.5));
      return kl_loss;
    });
  }

  // SIMPLIFIED loss function like RhythmVAE - no complex scheduling or masking
  vaeLoss(yTrue, yPred) {
    return tf.tidy(() => {
      const [yTrueOn, yTrueVel, yTrueDur, yTrueTime] = yTrue;
      const [z_mean, z_log_var, y] = yPred;
      const [yOn, yVel, yDur, yTime] = y;

      let onset_loss = this.reconstructionLoss(yTrueOn, yOn);
      onset_loss = onset_loss.mul(ON_LOSS_COEF);
      let velocity_loss = this.mseLoss(yTrueVel, yVel);
      velocity_loss = velocity_loss.mul(VEL_LOSS_COEF);
      let duration_loss = this.mseLoss(yTrueDur, yDur);
      duration_loss = duration_loss.mul(DUR_LOSS_COEF);
      let timeshift_loss = this.mseLoss(yTrueTime, yTime);
      timeshift_loss = timeshift_loss.mul(TIME_LOSS_COEF);

      const kl_loss = this.klLoss(z_mean, z_log_var);

      // Store individual losses for detailed logging
      this.lastLossBreakdown = {
        onset: tf.mean(onset_loss).dataSync()[0],
        velocity: tf.mean(velocity_loss).dataSync()[0],
        duration: tf.mean(duration_loss).dataSync()[0],
        timeshift: tf.mean(timeshift_loss).dataSync()[0],
        kl: tf.mean(kl_loss).dataSync()[0],
      };

      // Simple, direct loss combination like RhythmVAE
      const total_loss = tf.mean(
        onset_loss
          .add(velocity_loss)
          .add(duration_loss)
          .add(timeshift_loss)
          .add(kl_loss)
      );
      return total_loss;
    });
  }

  async train(trainConfig) {
    this.isTrained = false;
    this.isTraining = true;
    this.shouldStopTraining = false;
    if (trainConfig != undefined) {
      this.trainConfig = trainConfig;
    }
    const config = this.trainConfig;

    const batchSize = config.batchSize;
    const numBatch = Math.floor(dataHandlerOnset.getDataSize() / batchSize);
    const epochs = numEpochs;
    const testBatchSize = config.testBatchSize;
    const optimizer = config.optimizer;
    const logMessage = console.log;

    const originalDim = this.modelConfig.originalDim;

    // Performance optimization: Enable mixed precision for M1 Max
    tf.env().set("WEBGL_FORCE_F16_TEXTURES", true);
    tf.env().set("WEBGL_PACK", true);

    Max.outlet("training", 1);
    utils.log_status(
      "🎵 Starting MelodyVAE training with timeshift learning..."
    );

    for (let i = 0; i < epochs; i++) {
      if (this.shouldStopTraining) break;

      let batchInputOn, batchInputVel, batchInputDur, batchInputTime;
      let testBatchInputOn,
        testBatchInputVel,
        testBatchInputDur,
        testBatchInputTime;
      let trainLoss; // for a training batch
      let epochLoss, valLoss;

      logMessage(`[Epoch ${i + 1}]\n`);
      Max.outlet("epoch", i + 1, epochs);
      epochLoss = 0;

      // Training
      for (let j = 0; j < numBatch; j++) {
        batchInputOn = dataHandlerOnset
          .nextTrainBatch(batchSize)
          .xs.reshape([batchSize, originalDim]);
        batchInputVel = dataHandlerVelocity
          .nextTrainBatch(batchSize)
          .xs.reshape([batchSize, originalDim]);
        batchInputDur = dataHandlerDuration
          .nextTrainBatch(batchSize)
          .xs.reshape([batchSize, originalDim]);
        batchInputTime = dataHandlerTimeshift
          .nextTrainBatch(batchSize)
          .xs.reshape([batchSize, originalDim]);
        trainLoss = await optimizer.minimize(
          () =>
            this.vaeLoss(
              [batchInputOn, batchInputVel, batchInputDur, batchInputTime],
              this.apply([
                batchInputOn,
                batchInputVel,
                batchInputDur,
                batchInputTime,
              ])
            ),
          true
        );
        trainLoss = Number(trainLoss.dataSync());
        epochLoss = epochLoss + trainLoss;

        await tf.nextFrame();
      }
      epochLoss = epochLoss / numBatch; // average
      logMessage(
        `\t[Average] Training Loss: ${epochLoss.toFixed(
          3
        )}. Epoch ${i} / ${epochs} \n`
      );
      Max.outlet("loss", epochLoss);

      // Validation
      testBatchInputOn = dataHandlerOnset
        .nextTestBatch(testBatchSize)
        .xs.reshape([testBatchSize, originalDim]);
      testBatchInputVel = dataHandlerVelocity
        .nextTestBatch(testBatchSize)
        .xs.reshape([testBatchSize, originalDim]);
      testBatchInputDur = dataHandlerDuration
        .nextTestBatch(testBatchSize)
        .xs.reshape([testBatchSize, originalDim]);
      testBatchInputTime = dataHandlerTimeshift
        .nextTestBatch(testBatchSize)
        .xs.reshape([testBatchSize, originalDim]);
      valLoss = this.vaeLoss(
        [
          testBatchInputOn,
          testBatchInputVel,
          testBatchInputDur,
          testBatchInputTime,
        ],
        this.apply([
          testBatchInputOn,
          testBatchInputVel,
          testBatchInputDur,
          testBatchInputTime,
        ])
      );
      valLoss = Number(valLoss.dataSync());

      logMessage(`\tVal Loss: ${valLoss.toFixed(3)}. Epoch ${i} / ${epochs}\n`);
      Max.outlet("val_loss", valLoss);

      await tf.nextFrame();
    }
    this.isTrained = true;
    this.isTraining = false;
    Max.outlet("training", 0);
    utils.log_status(
      "🎉 Training finished! Timeshift patterns learned successfully."
    );
  }

  generate(zs) {
    let [outputsOn, outputsVel, outputsDur, outputsTime] =
      this.decoder.apply(zs);

    outputsOn = outputsOn.reshape([NUM_MIDI_CLASSES, LOOP_DURATION]);
    outputsVel = outputsVel.reshape([NUM_MIDI_CLASSES, LOOP_DURATION]);
    outputsDur = outputsDur.reshape([NUM_MIDI_CLASSES, LOOP_DURATION]);
    outputsTime = outputsTime.reshape([NUM_MIDI_CLASSES, LOOP_DURATION]);

    return [
      outputsOn.arraySync(),
      outputsVel.arraySync(),
      outputsDur.arraySync(),
      outputsTime.arraySync(),
    ];
  }

  bendModel(noise_range) {
    let weights = [];
    for (let i = 0; i < this.decoder.getWeights().length; i++) {
      let w = this.decoder.getWeights()[i];
      let shape = w.shape;
      console.log(shape);
      let noise = tf.randomNormal(w.shape, 0.0, noise_range);
      let neww = tf.add(w, noise);
      weights.push(neww);
    }
    this.decoder.setWeights(weights);
  }

  async saveModel(path) {
    const saved = await this.decoder.save(path);
    utils.post(saved);
  }

  async loadModel(path) {
    this.decoder = await tf.loadLayersModel(path);
    this.isTrained = true;
  }

  encode(inputOn, inputVel, inputDur, inputTime) {
    if (!this.encoder) {
      utils.error_status("Model is not trained yet");
      return;
    }

    // reshaping...
    inputOn = inputOn.reshape([1, ORIGINAL_DIM]);
    inputVel = inputVel.reshape([1, ORIGINAL_DIM]);
    inputDur = inputDur.reshape([1, ORIGINAL_DIM]);
    inputTime = inputTime.reshape([1, ORIGINAL_DIM]);

    let [zMean, zLogVar, zs] = this.encoder.apply([
      inputOn,
      inputVel,
      inputDur,
      inputTime,
    ]);
    this.generate(zs); // generate melody pattern with the encoded z
    zs = zs.arraySync();
    return zs[0];
  }
}

exports.loadAndTrain = loadAndTrain;
exports.saveModel = saveModel;
exports.loadModel = loadModel;
exports.clearModel = clearModel;
exports.generatePattern = generatePattern;
exports.encodePattern = encodePattern;
exports.stopTraining = stopTraining;
exports.isReadyToGenerate = isReadyToGenerate;
exports.isTraining = isTraining;
exports.setEpochs = setEpochs;
exports.bendModel = bendModel;
