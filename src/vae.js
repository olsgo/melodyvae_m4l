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
const INTERMEDIATE_DIM = 512;
const LATENT_DIM = 2;

const BATCH_SIZE = 64; // Reduced from 128 to match RhythmVAE for better convergence
const TEST_BATCH_SIZE = 128; // Reduced from 1000 for more representative validation
const ON_LOSS_COEF = 0.5; // Reduced to give timeshift more relative weight
const DUR_LOSS_COEF = 0.5; // Reduced from 1.0
const VEL_LOSS_COEF = 1.0; // Reduced from 2.5
const TIME_LOSS_COEF = 100.0; // Increased from 50.0 but not extreme

// Training optimization parameters
const LEARNING_RATE = 0.001; // Initial learning rate
const LEARNING_RATE_DECAY = 0.98; // More conservative decay factor
const LEARNING_RATE_DECAY_EPOCHS = 8; // Decay every 8 epochs instead of 5
const GRADIENT_CLIP_NORM = 1.0; // Gradient clipping threshold
const KL_ANNEALING_EPOCHS = 60; // Even slower KL annealing
const TIMESHIFT_RAMP_EPOCHS = 5; // Faster timeshift ramp-up
const DROPOUT_RATE = 0.2; // Dropout rate for regularization
const EARLY_STOPPING_PATIENCE = 35; // More patience for complex timeshift learning (increased from 25)

// Define the optimizer with gradient clipping
const optimizer = tf.train.adam({
  learningRate: LEARNING_RATE,
  clipNorm: GRADIENT_CLIP_NORM,
});

let dataHandlerOnset;
let dataHandlerVelocity;
let dataHandlerDuration;
let dataHandlerTimeshift;
let model;
let numEpochs = 100;
let currentLearningRate = LEARNING_RATE;
let bestValLoss = Infinity;
let epochsWithoutImprovement = 0;

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
  ); // data utility fo onset
  dataHandlerVelocity = new data.DataHandler(
    train_data_velocity,
    train_indices,
    test_indices
  ); // data utility for velocity
  dataHandlerDuration = new data.DataHandler(
    train_data_duration,
    train_indices,
    test_indices
  ); // data utility for duration
  dataHandlerTimeshift = new data.DataHandler(
    train_data_timeshift,
    train_indices,
    test_indices
  ); // data utility for timeshift

  // start training!
  initModel(); // initializing model class
  startTraining(); // start the actual training process with the given training data
}

function initModel() {
  // Reset training state
  currentLearningRate = LEARNING_RATE;
  bestValLoss = Infinity;
  epochsWithoutImprovement = 0;

  model = new ConditionalVAE({
    modelConfig: {
      originalDim: ORIGINAL_DIM,
      intermediateDim: INTERMEDIATE_DIM,
      latentDim: LATENT_DIM,
    },
    trainConfig: {
      batchSize: BATCH_SIZE,
      testBatchSize: TEST_BATCH_SIZE,
      optimizer: tf.train.adam(currentLearningRate),
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
  // return (model && model.isTrained);
  return model;
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

// Add a class variable to track variance issues
class ConditionalVAE {
  constructor(config) {
    this.modelConfig = config.modelConfig;
    this.trainConfig = config.trainConfig;
    [this.encoder, this.decoder, this.apply] = this.build();
    this.isTrained = false;
    this.timeshiftVarianceBoost = 1.0; // New variable to track boost factor
  }

  build(modelConfig) {
    if (modelConfig != undefined) {
      this.modelConfig = modelConfig;
    }
    const config = this.modelConfig;

    const originalDim = config.originalDim;
    const intermediateDim = config.intermediateDim;
    const latentDim = config.latentDim;

    // Define local constants to resolve ReferenceError
    const z_dim = latentDim;
    const original_dim = originalDim;

    // VAE model = encoder + decoder
    // build encoder model

    // Onset Input
    const encoderInputsOn = tf.input({ shape: [originalDim] });
    const x1LinearOn = tf.layers
      .dense({
        units: intermediateDim,
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(encoderInputsOn);
    const x1NormalisedOn = tf.layers
      .batchNormalization({ axis: 1 })
      .apply(x1LinearOn);
    const x1OnPreDropout = tf.layers.leakyReLU().apply(x1NormalisedOn);
    const x1On = tf.layers
      .dropout({ rate: DROPOUT_RATE })
      .apply(x1OnPreDropout);

    // Velocity input
    const encoderInputsVel = tf.input({ shape: [originalDim] });
    const x1LinearVel = tf.layers
      .dense({
        units: intermediateDim,
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(encoderInputsVel);
    const x1NormalisedVel = tf.layers
      .batchNormalization({ axis: 1 })
      .apply(x1LinearVel);
    const x1VelPreDropout = tf.layers.leakyReLU().apply(x1NormalisedVel);
    const x1Vel = tf.layers
      .dropout({ rate: DROPOUT_RATE })
      .apply(x1VelPreDropout);

    // Duration input
    const encoderInputsDur = tf.input({ shape: [originalDim] });
    const x1LinearDur = tf.layers
      .dense({
        units: intermediateDim,
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(encoderInputsDur);
    const x1NormalisedDur = tf.layers
      .batchNormalization({ axis: 1 })
      .apply(x1LinearDur);
    const x1DurPreDropout = tf.layers.leakyReLU().apply(x1NormalisedDur);
    const x1Dur = tf.layers
      .dropout({ rate: DROPOUT_RATE })
      .apply(x1DurPreDropout);

    // Timeshift input
    const encoderInputsTime = tf.input({ shape: [originalDim] });
    const x1LinearTime = tf.layers
      .dense({
        units: intermediateDim,
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(encoderInputsTime);
    const x1NormalisedTime = tf.layers
      .batchNormalization({ axis: 1 })
      .apply(x1LinearTime);
    const x1TimePreDropout = tf.layers.leakyReLU().apply(x1NormalisedTime);
    const x1Time = tf.layers
      .dropout({ rate: DROPOUT_RATE })
      .apply(x1TimePreDropout);

    // Merged
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
    const x2PreDropout = tf.layers.leakyReLU().apply(x2Normalised);
    const x2 = tf.layers.dropout({ rate: DROPOUT_RATE }).apply(x2PreDropout);

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

    // build decoder model
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
    const x3PreDropout = tf.layers.leakyReLU().apply(x3Normalised);
    const x3 = tf.layers.dropout({ rate: DROPOUT_RATE }).apply(x3PreDropout);

    // Decoder for onsets
    const x4LinearOn = tf.layers
      .dense({
        units: intermediateDim,
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(x3);
    const x4NormalisedOn = tf.layers
      .batchNormalization({ axis: 1 })
      .apply(x4LinearOn);
    const x4OnPreDropout = tf.layers.leakyReLU().apply(x4NormalisedOn);
    const x4On = tf.layers
      .dropout({ rate: DROPOUT_RATE })
      .apply(x4OnPreDropout);
    const decoderOutputsOn = tf.layers
      .dense({ units: originalDim, activation: "sigmoid" })
      .apply(x4On);

    // Decoder for velocity
    const x4LinearVel = tf.layers
      .dense({
        units: intermediateDim,
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(x3);
    const x4NormalisedVel = tf.layers
      .batchNormalization({ axis: 1 })
      .apply(x4LinearVel);
    const x4VelPreDropout = tf.layers.leakyReLU().apply(x4NormalisedVel);
    const x4Vel = tf.layers
      .dropout({ rate: DROPOUT_RATE })
      .apply(x4VelPreDropout);
    const decoderOutputsVel = tf.layers
      .dense({ units: originalDim, activation: "sigmoid" })
      .apply(x4Vel);

    // Decoder for duration
    const x4LinearDur = tf.layers
      .dense({
        units: intermediateDim,
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(x3);
    const x4NormalisedDur = tf.layers
      .batchNormalization({ axis: 1 })
      .apply(x4LinearDur);
    const x4DurPreDropout = tf.layers.leakyReLU().apply(x4NormalisedDur);
    const x4Dur = tf.layers
      .dropout({ rate: DROPOUT_RATE })
      .apply(x4DurPreDropout);
    const decoderOutputsDur = tf.layers
      .dense({ units: originalDim, activation: "relu" })
      .apply(x4Dur);

    // Decoder for timeshift - FIXED ARCHITECTURE
    const x4LinearTime = tf.layers
      .dense({
        units: intermediateDim * 2, // Increased capacity
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(x3);
    const x4NormalisedTime = tf.layers
      .batchNormalization({ axis: 1 })
      .apply(x4LinearTime);
    const x4TimePreDropout = tf.layers.leakyReLU().apply(x4NormalisedTime);
    const x4Time = tf.layers
      .dropout({ rate: DROPOUT_RATE })
      .apply(x4TimePreDropout);

    // Add an additional layer for timeshift complexity
    const x5LinearTime = tf.layers
      .dense({
        units: intermediateDim / 2,
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(x4Time);
    const x5NormalisedTime = tf.layers
      .batchNormalization({ axis: 1 })
      .apply(x5LinearTime);
    const x5TimePreDropout = tf.layers.leakyReLU().apply(x5NormalisedTime);
    const x5Time = tf.layers
      .dropout({ rate: DROPOUT_RATE })
      .apply(x5TimePreDropout);

    // Add a third layer specifically for timeshift
    const x6LinearTime = tf.layers
      .dense({
        units: intermediateDim,
        useBias: true,
        kernelInitializer: "glorotNormal",
      })
      .apply(x5Time);
    const x6NormalisedTime = tf.layers
      .batchNormalization({ axis: 1 })
      .apply(x6LinearTime);
    const x6TimePreDropout = tf.layers.leakyReLU().apply(x6NormalisedTime);
    const x6Time = tf.layers
      .dropout({ rate: DROPOUT_RATE * 0.5 }) // Less dropout for timeshift
      .apply(x6TimePreDropout);

    // Final timeshift output
    const decoderOutputsTime = tf.layers
      .dense({ units: originalDim, activation: "tanh" })
      .apply(x6Time);

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

  maeLoss(yTrue, yPred) {
    return tf.tidy(() => {
      let mae_loss = tf.metrics.meanAbsoluteError(yTrue, yPred);
      mae_loss = mae_loss.mul(tf.scalar(yPred.shape[1]));
      return mae_loss;
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

  timeshiftDiversityLoss(yTime) {
    return tf.tidy(() => {
      // Encourage outputs to use the full tanh range [-1, 1]
      const batchSize = yTime.shape[0];
      const sequenceLength = yTime.shape[1];

      // Flatten timeshift outputs
      const flatTimeshift = yTime.reshape([batchSize * sequenceLength]);

      // Calculate pairwise distances to encourage diversity
      const expanded1 = flatTimeshift.expandDims(1);
      const expanded2 = flatTimeshift.expandDims(0);
      const distances = tf.abs(expanded1.sub(expanded2));

      // Penalize when outputs are too similar
      const diversityTarget = tf.scalar(0.1);
      const diversityPenalty = diversityTarget.sub(tf.mean(distances)).relu();

      return diversityPenalty.mul(tf.scalar(50.0));
    });
  }

  vaeLoss(yTrue, yPred, currentEpoch = 0) {
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

      // More gradual timeshift loss ramping to prevent early domination
      const mask = yTrueOn.greater(0.5).cast("float32"); // 1 where a note exists
      const maskedTrue = yTrueTime.mul(mask);
      const maskedPred = yTime.mul(mask);

      // normalise so each note, not each grid-cell, has equal weight
      const denom = tf.maximum(tf.sum(mask), 1);
      let timeshift_loss = tf.losses
        .huberLoss(maskedTrue, maskedPred)
        .sum()
        .div(denom);
      const timeshift_weight = Math.min(
        1.0,
        (currentEpoch + 1) / TIMESHIFT_RAMP_EPOCHS
      );

      // Add variance regularization to prevent collapse - SAFE VERSION
      const timeshiftMoments = tf.moments(maskedPred);
      const currentVariance = timeshiftMoments.variance;

      // Safe variance penalty - clamped to prevent explosion
      const targetVariance = 0.02; // Reasonable target
      const varianceDeficit = Math.max(
        0,
        targetVariance - currentVariance.dataSync()[0]
      );
      const variancePenalty = tf.scalar(varianceDeficit * 100.0); // Moderate penalty

      timeshift_loss = timeshift_loss.add(variancePenalty);

      // Add diversity loss to encourage wider output range
      const diversityLoss = this.timeshiftDiversityLoss(maskedPred);
      timeshift_loss = timeshift_loss.add(diversityLoss);

      // Cleanup
      timeshiftMoments.mean.dispose();

      // Apply the variance recovery boost if needed
      timeshift_loss = timeshift_loss.mul(
        TIME_LOSS_COEF * this.timeshiftVarianceBoost
      );

      // Timeshift focus mode for first 20 epochs
      if (currentEpoch < 20) {
        // Temporarily reduce other loss components
        onset_loss = onset_loss.mul(tf.scalar(0.1));
        velocity_loss = velocity_loss.mul(tf.scalar(0.1));
        duration_loss = duration_loss.mul(tf.scalar(0.1));

        // Boost timeshift learning even more
        const timeshift_boost = 5.0;
        timeshift_loss = timeshift_loss.mul(tf.scalar(timeshift_boost));
      }

      // Much slower KL annealing (β-VAE technique)
      const kl_loss = this.klLoss(z_mean, z_log_var);
      // Start KL annealing much later to let timeshift learn first
      const kl_weight =
        currentEpoch > 30
          ? Math.min(1.0, (currentEpoch - 30) / KL_ANNEALING_EPOCHS)
          : 0.0;
      const weighted_kl_loss = kl_loss.mul(tf.scalar(kl_weight));

      // Reset boost factor after applying it
      this.timeshiftVarianceBoost = 1.0;

      const total_loss = tf.mean(
        onset_loss
          .add(velocity_loss)
          .add(duration_loss)
          .add(timeshift_loss)
          .add(weighted_kl_loss)
      );
      return total_loss;
    });
  }

  async train(data, trainConfig) {
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
    let optimizer = tf.train.adam(LEARNING_RATE, ADAM_BETA1, ADAM_BETA2);
    let timeshiftOptimizer = tf.train.adam(
      LEARNING_RATE * 0.1,
      ADAM_BETA1,
      ADAM_BETA2
    ); // Lower LR for timeshift head
    const logMessage = utils.post; // Use utils.post for logging

    const originalDim = this.modelConfig.originalDim;

    // Reset training state
    bestValLoss = Infinity;
    epochsWithoutImprovement = 0;
    currentLearningRate = LEARNING_RATE;

    // Initialize timeshift variance tracking
    this.timeshiftVarianceHistory = [];
    this.timeshiftStatsHistory = [];
    this.timeshiftVarianceBoost = 1.0;
    let consecutiveLowVarianceEpochs = 0;

    Max.outlet("training", 1);
    for (let i = 0; i < epochs; i++) {
      if (this.shouldStopTraining) break;

      let batchInputOn, batchInputVel, batchInputDur, batchInputTime;
      let testBatchInputOn,
        testBatchInputVel,
        testBatchInputDur,
        testBatchInputTime;
      let trainLoss;
      let epochLoss, valLoss;

      logMessage(`[Epoch ${i + 1}]\n`);
      Max.outlet("epoch", i + 1, epochs);

      // Calculate current weight schedules for logging
      const current_kl_weight = Math.min(1.0, i / KL_ANNEALING_EPOCHS);
      const current_timeshift_weight = Math.min(
        1.0,
        (i + 1) / TIMESHIFT_RAMP_EPOCHS
      );

      utils.log_status(
        `Epoch: ${i + 1} (LR: ${currentLearningRate.toFixed(
          6
        )}, KL: ${current_kl_weight.toFixed(
          3
        )}, TS: ${current_timeshift_weight.toFixed(3)})`
      );

      epochLoss = 0;

      // Learning rate decay - less aggressive schedule
      if (i > 0 && i % LEARNING_RATE_DECAY_EPOCHS === 0) {
        currentLearningRate *= LEARNING_RATE_DECAY;
        optimizer = tf.train.adam(currentLearningRate);
        timeshiftOptimizer = tf.train.adam(currentLearningRate * 0.1); // Also decay the timeshift optimizer
        logMessage(
          `\tLearning rate decayed to: ${currentLearningRate.toFixed(6)}`
        );

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

        // Training step with gradient clipping and logging
        const f = () => {
          const [zMean, zLogVar, outputs] = this.apply([
            batchInputOn,
            batchInputVel,
            batchInputDur,
            batchInputTime,
          ]);

          return this.vaeLoss(
            [batchInputOn, batchInputVel, batchInputDur, batchInputTime],
            [zMean, zLogVar, outputs],
            i
          );
        };

      const { value, grads } = tf.variableGrads(
          f,
          this.decoder.getWeights(true)
        );

        // Separate gradients for timeshift head
        const timeshiftVars = this.decoder.layers.slice(-6).flatMap(l => l.getWeights());
        const timeshiftGrads = {};
        const mainGrads = {};

        for (const gradName in grads) {
          if (timeshiftVars.find(v => v.name === gradName)) {
            timeshiftGrads[gradName] = grads[gradName];
          } else {
            mainGrads[gradName] = grads[gradName];
          }
        }

        // Clip and apply gradients
        const clippedMainGrads = Object.keys(mainGrads).reduce((acc, key) => {
            acc[key] = tf.clipByValue(mainGrads[key], -5.0, 5.0);
            return acc;
        }, {});
        const clippedTimeshiftGrads = Object.keys(timeshiftGrads).reduce((acc, key) => {
            acc[key] = tf.clipByValue(timeshiftGrads[key], -5.0, 5.0);
            return acc;
        }, {});

        optimizer.applyGradients(clippedMainGrads);
        timeshiftOptimizer.applyGradients(clippedTimeshiftGrads);

        const timeshiftDecoderLayer =
          this.decoder.layers[this.decoder.layers.length - 1];
          
        const timeshiftGrads = grads[timeshiftDecoderLayer.name];
        if (timeshiftGrads) {
          logMessage(
            `	   Timeshift Grads Mean: ${tf
              .mean(timeshiftGrads)
              .dataSync()[0]
              .toFixed(6)}, Max: ${tf
              .max(timeshiftGrads)
              .dataSync()[0]
              .toFixed(6)}`
          );
        }

        // Dispose original and clipped grads to prevent memory leaks
        for (const key in grads) {
          grads[key].dispose();
        }

        trainLoss = Number(value.dataSync());
        epochLoss = epochLoss + trainLoss;

        await tf.nextFrame();
      }
      epochLoss = epochLoss / numBatch; // average
      logMessage(
        `\t[Average] Training Loss: ${epochLoss.toFixed(
          3
        )}. Epoch ${i} / ${epochs} \n`
      );

      // Log timeshift focus mode
      if (i < 20) {
        logMessage(
          `\t   🎯 TIMESHIFT FOCUS MODE: Boosting timeshift learning by 5.0x`
        );
      }
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
        ]),
        i
      );
      valLoss = Number(valLoss.dataSync());

      logMessage(`\tVal Loss: ${valLoss.toFixed(3)}. Epoch ${i} / ${epochs}\n`);
      Max.outlet("val_loss", valLoss);

      // In the train() method, replace the existing timeshift variance monitoring code:

      // Monitor timeshift output variance to detect collapse
      const [, , outputs] = this.apply([
        testBatchInputOn,
        testBatchInputVel,
        testBatchInputDur,
        testBatchInputTime,
      ]);
      const decodedTimeshift = outputs[3];

      if (decodedTimeshift != null && i % 5 === 0) {
        // Only check every 5 epochs
        const mask = testBatchInputOn.greater(0.5).cast("float32");
        const maskedPred = decodedTimeshift.mul(mask);
        const maskedVariance = tf.moments(maskedPred).variance.dataSync()[0];

        if (maskedVariance < 0.005) {
          // still collapsing
          this.timeshiftVarianceBoost = Math.min(
            this.timeshiftVarianceBoost * 2,
            8
          );
        } else {
          this.timeshiftVarianceBoost = 1.0; // back to normal once variance recovers
        }

        const timeshiftVariance = tf
          .moments(decodedTimeshift)
          .variance.dataSync()[0];
        const timeshiftMean = tf.mean(decodedTimeshift).dataSync()[0];
        const timeshiftMin = tf.min(decodedTimeshift).dataSync()[0];
        const timeshiftMax = tf.max(decodedTimeshift).dataSync()[0];

        logMessage(
          `\t📊 TIMESHIFT: Var=${timeshiftVariance.toFixed(
            6
          )}, Range=[${timeshiftMin.toFixed(3)}, ${timeshiftMax.toFixed(3)}]`
        );

        if (timeshiftVariance < 0.001) {
          logMessage(
            `\t⚠️  TIMESHIFT COLLAPSE DETECTED - Variance: ${timeshiftVariance.toFixed(
              6
            )}`
          );
        }
      }

      // Early stopping check
      if (valLoss < bestValLoss) {
        bestValLoss = valLoss;
        epochsWithoutImprovement = 0;
        logMessage(`\tNew best validation loss: ${bestValLoss.toFixed(3)}`);
      } else {
        epochsWithoutImprovement++;
        if (epochsWithoutImprovement >= EARLY_STOPPING_PATIENCE) {
          logMessage(
            `\tEarly stopping: no improvement for ${EARLY_STOPPING_PATIENCE} epochs`
          );
          utils.log_status(`Early stopping at epoch ${i + 1}`);
          break;
        }
      }

      await tf.nextFrame();
    }

    // Final timeshift learning summary
    if (this.timeshiftVarianceHistory.length > 0) {
      const finalVariance =
        this.timeshiftVarianceHistory[this.timeshiftVarianceHistory.length - 1];
      const maxVariance = Math.max(...this.timeshiftVarianceHistory);
      const avgVariance =
        this.timeshiftVarianceHistory.reduce((a, b) => a + b) /
        this.timeshiftVarianceHistory.length;

      logMessage(`\n🎯 TIMESHIFT LEARNING SUMMARY:`);
      logMessage(`   Final variance: ${finalVariance.toFixed(6)}`);
      logMessage(`   Peak variance: ${maxVariance.toFixed(6)}`);
      logMessage(`   Average variance: ${avgVariance.toFixed(6)}`);
      logMessage(
        `   Learning assessment: ${
          finalVariance > 0.02
            ? "GOOD - Model learned timing variations"
            : finalVariance > 0.01
            ? "MODERATE - Some timing learning"
            : "POOR - Limited timing learning"
        }`
      );
    }

    this.isTrained = true;
    this.isTraining = false;
    Max.outlet("training", 0);
    utils.log_status("Training finished!");
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

function range(start, edge, step) {
  // If only one number was passed in make it the edge and 0 the start.
  if (arguments.length == 1) {
    edge = start;
    start = 0;
  }

  // Validate the edge and step numbers.
  edge = edge || 0;
  step = step || 1;

  // Create the array of numbers, stopping befor the edge.
  for (var ret = []; (edge - start) * step > 0; start += step) {
    ret.push(start);
  }
  return ret;
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
