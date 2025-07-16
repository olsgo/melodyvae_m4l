// VAE in tensorflow.js
// based on https://github.com/songer1993/tfjs-vae

const Max = require('max-api');
const tf = require('@tensorflow/tfjs-node');

const utils = require('./utils.js')
const data = require('./data.js')

// Constants
const NUM_MIDI_CLASSES = require('./constants.js').NUM_MIDI_CLASSES;
const LOOP_DURATION = require('./constants.js').LOOP_DURATION;

const ORIGINAL_DIM = require('./constants.js').ORIGINAL_DIM;
const INTERMEDIATE_DIM = 512;
const LATENT_DIM = 2;

const BATCH_SIZE = 64;  // Reduced from 128 to match RhythmVAE for better convergence
const NUM_BATCH = 50;
const TEST_BATCH_SIZE = 128;  // Reduced from 1000 for more representative validation
const ON_LOSS_COEF = 1.0;  // Increased from 0.75 for better onset reconstruction
const DUR_LOSS_COEF = 1.0;  // coef for duration loss
const VEL_LOSS_COEF = 2.5;  // coef for velocity loss
const TIME_LOSS_COEF = 2.0;  // Increased from 1.5 for better timeshift learning

// Training optimization parameters
const LEARNING_RATE = 0.001;  // Initial learning rate
const LEARNING_RATE_DECAY = 0.98;  // More conservative decay factor
const LEARNING_RATE_DECAY_EPOCHS = 8;  // Decay every 8 epochs instead of 5
const GRADIENT_CLIP_NORM = 1.0;  // Gradient clipping threshold
const KL_ANNEALING_EPOCHS = 20;  // Slower KL annealing to prevent timeshift collapse
const TIMESHIFT_RAMP_EPOCHS = 15;  // Gradual ramping of timeshift loss weight
const DROPOUT_RATE = 0.2;  // Dropout rate for regularization
const EARLY_STOPPING_PATIENCE = 25;  // More patience for complex timeshift learning

let dataHandlerOnset;
let dataHandlerVelocity;
let dataHandlerDuration;
let dataHandlerTimeshift;
let model;
let numEpochs = 100;
let currentLearningRate = LEARNING_RATE;
let bestValLoss = Infinity;
let epochsWithoutImprovement = 0;

async function loadAndTrain(train_data_onset, train_data_velocity, train_data_duration, train_data_timeshift) {
  console.assert(train_data_onset.length == train_data_velocity.length && 
                 train_data_velocity.length == train_data_duration.length &&
                 train_data_duration.length == train_data_timeshift.length);
  
  // shuffle in sync
  const total_num = train_data_onset.length;
  shuffled_indices = tf.util.createShuffledIndices(total_num);
  train_data_onset = utils.shuffle_with_indices(train_data_onset,shuffled_indices);
  train_data_velocity = utils.shuffle_with_indices(train_data_velocity,shuffled_indices);
  train_data_duration = utils.shuffle_with_indices(train_data_duration,shuffled_indices);
  train_data_timeshift = utils.shuffle_with_indices(train_data_timeshift,shuffled_indices);

  // synced indices
  const num_trains = Math.floor(data.TRAIN_TEST_RATIO * total_num);
  const num_tests  = total_num - num_trains;
  const train_indices = tf.util.createShuffledIndices(num_trains);
  const test_indices = tf.util.createShuffledIndices(num_tests);

  // create data handlers
  dataHandlerOnset = new data.DataHandler(train_data_onset, train_indices, test_indices); // data utility fo onset
  dataHandlerVelocity = new data.DataHandler(train_data_velocity, train_indices, test_indices); // data utility for velocity
  dataHandlerDuration = new data.DataHandler(train_data_duration, train_indices, test_indices); // data utility for duration
  dataHandlerTimeshift = new data.DataHandler(train_data_timeshift, train_indices, test_indices); // data utility for timeshift

  // start training!
  initModel(); // initializing model class
  startTraining(); // start the actual training process with the given training data
}

function initModel(){
  // Reset training state
  currentLearningRate = LEARNING_RATE;
  bestValLoss = Infinity;
  epochsWithoutImprovement = 0;
  
  model = new ConditionalVAE({
    modelConfig:{
      originalDim: ORIGINAL_DIM,
      intermediateDim: INTERMEDIATE_DIM,
      latentDim: LATENT_DIM
    },
    trainConfig:{
      batchSize: BATCH_SIZE,
      testBatchSize: TEST_BATCH_SIZE,
      optimizer: tf.train.adam(currentLearningRate)
    }
  });
}

async function startTraining(){
  await model.train();
}

function stopTraining(){
  model.shouldStopTraining = true;
  utils.log_status("Stopping training...");
}

function isTraining(){
  if (model && model.isTraining) return true;
}

function isReadyToGenerate(){
  // return (model && model.isTrained);
  return (model);
}

function setEpochs(e){
  numEpochs = e;
  Max.outlet("epoch", 0, numEpochs);
}

function generatePattern(z1, z2, noise_range=0.0){
  var zs;
  if (z1 === 'undefined' || z2 === 'undefined'){
    zs = tf.randomNormal([1, 2]);
  } else {
    zs = tf.tensor2d([[z1, z2]]);
  }

  // noise
  if (noise_range > 0.0){
    var noise = tf.randomNormal([1, 2]);
    zs = zs.add(noise.mul(tf.scalar(noise_range)));
  }
  return model.generate(zs);
}

function encodePattern(inputOn, inputVel, inputDur, inputTime){
  return model.encode(inputOn, inputVel, inputDur, inputTime);
}

function clearModel(){
  model = null;
}

function bendModel(noise_range){
  model.bendModel(noise_range);
}

async function saveModel(filepath){
  model.saveModel(filepath);
}

async function loadModel(filepath){
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
    return 'sampleLayer';
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
    if (modelConfig != undefined){
      this.modelConfig = modelConfig;
    }
    const config = this.modelConfig;

    const originalDim = config.originalDim;
    const intermediateDim = config.intermediateDim;
    const latentDim = config.latentDim;

    // VAE model = encoder + decoder
    // build encoder model

    // Onset Input
    const encoderInputsOn = tf.input({shape: [originalDim]});
    const x1LinearOn = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(encoderInputsOn);
    const x1NormalisedOn = tf.layers.batchNormalization({axis: 1}).apply(x1LinearOn);
    const x1OnPreDropout = tf.layers.leakyReLU().apply(x1NormalisedOn);
    const x1On = tf.layers.dropout({rate: DROPOUT_RATE}).apply(x1OnPreDropout);

    // Velocity input
    const encoderInputsVel = tf.input({shape: [originalDim]});
    const x1LinearVel = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(encoderInputsVel);
    const x1NormalisedVel = tf.layers.batchNormalization({axis: 1}).apply(x1LinearVel);
    const x1VelPreDropout = tf.layers.leakyReLU().apply(x1NormalisedVel);
    const x1Vel = tf.layers.dropout({rate: DROPOUT_RATE}).apply(x1VelPreDropout);
    
    // Duration input
    const encoderInputsDur = tf.input({shape: [originalDim]});
    const x1LinearDur = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(encoderInputsDur);
    const x1NormalisedDur = tf.layers.batchNormalization({axis: 1}).apply(x1LinearDur);
    const x1DurPreDropout = tf.layers.leakyReLU().apply(x1NormalisedDur);
    const x1Dur = tf.layers.dropout({rate: DROPOUT_RATE}).apply(x1DurPreDropout);

    // Timeshift input
    const encoderInputsTime = tf.input({shape: [originalDim]});
    const x1LinearTime = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(encoderInputsTime);
    const x1NormalisedTime = tf.layers.batchNormalization({axis: 1}).apply(x1LinearTime);
    const x1TimePreDropout = tf.layers.leakyReLU().apply(x1NormalisedTime);
    const x1Time = tf.layers.dropout({rate: DROPOUT_RATE}).apply(x1TimePreDropout);

    // Merged
    const concatLayer = tf.layers.concatenate();
    const x1Merged = concatLayer.apply([x1On, x1Vel, x1Dur, x1Time]);
    const x2Linear = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x1Merged);
    const x2Normalised = tf.layers.batchNormalization({axis: 1}).apply(x2Linear);
    const x2PreDropout = tf.layers.leakyReLU().apply(x2Normalised);
    const x2 = tf.layers.dropout({rate: DROPOUT_RATE}).apply(x2PreDropout);
      
    const zMean = tf.layers.dense({units: latentDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x2);
    const zLogVar = tf.layers.dense({units: latentDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x2);
    const z = new sampleLayer().apply([zMean, zLogVar]);
    const encoderInputs = [encoderInputsOn, encoderInputsVel, encoderInputsDur, encoderInputsTime];
    const encoderOutputs = [zMean, zLogVar, z];

    const encoder = tf.model({inputs: encoderInputs, outputs: encoderOutputs, name: "encoder"})

    // build decoder model
    const decoderInputs = tf.input({shape: [latentDim]});
    const x3Linear = tf.layers.dense({units: intermediateDim * 2.0, useBias: true, kernelInitializer: 'glorotNormal'}).apply(decoderInputs);
    const x3Normalised = tf.layers.batchNormalization({axis: 1}).apply(x3Linear);
    const x3PreDropout = tf.layers.leakyReLU().apply(x3Normalised);
    const x3 = tf.layers.dropout({rate: DROPOUT_RATE}).apply(x3PreDropout);

    // Decoder for onsets
    const x4LinearOn = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x3);
    const x4NormalisedOn = tf.layers.batchNormalization({axis: 1}).apply(x4LinearOn);
    const x4OnPreDropout = tf.layers.leakyReLU().apply(x4NormalisedOn);
    const x4On = tf.layers.dropout({rate: DROPOUT_RATE}).apply(x4OnPreDropout);
    const decoderOutputsOn = tf.layers.dense({units: originalDim, activation: 'sigmoid'}).apply(x4On);

    // Decoder for velocity
    const x4LinearVel = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x3);
    const x4NormalisedVel = tf.layers.batchNormalization({axis: 1}).apply(x4LinearVel);
    const x4VelPreDropout = tf.layers.leakyReLU().apply(x4NormalisedVel);
    const x4Vel = tf.layers.dropout({rate: DROPOUT_RATE}).apply(x4VelPreDropout);
    const decoderOutputsVel = tf.layers.dense({units: originalDim, activation: 'sigmoid'}).apply(x4Vel);
    
    // Decoder for duration
    const x4LinearDur = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x3);
    const x4NormalisedDur = tf.layers.batchNormalization({axis: 1}).apply(x4LinearDur);
    const x4DurPreDropout = tf.layers.leakyReLU().apply(x4NormalisedDur);
    const x4Dur = tf.layers.dropout({rate: DROPOUT_RATE}).apply(x4DurPreDropout);
    const decoderOutputsDur = tf.layers.dense({units: originalDim, activation: 'relu'}).apply(x4Dur);

    // Decoder for timeshift
    const x4LinearTime = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x3);
    const x4NormalisedTime = tf.layers.batchNormalization({axis: 1}).apply(x4LinearTime);
    const x4TimePreDropout = tf.layers.leakyReLU().apply(x4NormalisedTime);
    const x4Time = tf.layers.dropout({rate: DROPOUT_RATE}).apply(x4TimePreDropout);
    const decoderOutputsTime = tf.layers.dense({units: originalDim, activation: 'tanh'}).apply(x4Time);  // tanh for -1 to +1 range

    const decoderOutputs = [decoderOutputsOn, decoderOutputsVel, decoderOutputsDur, decoderOutputsTime];

    // Decoder model
    const decoder = tf.model({inputs: decoderInputs, outputs: decoderOutputs, name: "decoder"})

    // build VAE model
    const vae = (inputs) => {
      return tf.tidy(() => {
        const [zMean, zLogVar, z] = this.encoder.apply(inputs);
        const outputs = this.decoder.apply(z);
        return [zMean, zLogVar, outputs];
      });
    }

    return [encoder, decoder, vae];
  }


  reconstructionLoss(yTrue, yPred) {
    return tf.tidy(() => {
      let reconstruction_loss;
      reconstruction_loss = tf.metrics.binaryCrossentropy(yTrue, yPred)
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
      kl_loss = tf.scalar(1).add(z_log_var).sub(z_mean.square()).sub(z_log_var.exp());
      kl_loss = tf.sum(kl_loss, -1);
      kl_loss = kl_loss.mul(tf.scalar(-0.5));
      return kl_loss;
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
      
      // Gradual timeshift loss ramping to prevent early domination
      let timeshift_loss = this.mseLoss(yTrueTime, yTime);
      const timeshift_weight = Math.min(1.0, (currentEpoch + 1) / TIMESHIFT_RAMP_EPOCHS);
      timeshift_loss = timeshift_loss.mul(TIME_LOSS_COEF * timeshift_weight);

      // KL loss with annealing (β-VAE technique)
      const kl_loss = this.klLoss(z_mean, z_log_var);
      const kl_weight = Math.min(1.0, currentEpoch / KL_ANNEALING_EPOCHS);
      const weighted_kl_loss = kl_loss.mul(tf.scalar(kl_weight));
      
      // console.log("onset_loss", tf.mean(onset_loss).dataSync());
      // console.log("velocity_loss", tf.mean(velocity_loss).dataSync());
      // console.log("duration_loss",  tf.mean(duration_loss).dataSync());
      // console.log("timeshift_loss",  tf.mean(timeshift_loss).dataSync());
      // console.log("kl_loss",  tf.mean(kl_loss).dataSync());
      // console.log("kl_weight", kl_weight);
      
      const total_loss = tf.mean(onset_loss.add(velocity_loss).add(duration_loss).add(timeshift_loss).add(weighted_kl_loss)); // averaged in the batch
      return total_loss;
    });
  }

  async train(data, trainConfig) {
    this.isTrained = false;
    this.isTraining = true;
    this.shouldStopTraining = false;
    if (trainConfig != undefined){
      this.trainConfig = trainConfig;
    }
    const config = this.trainConfig;

    const batchSize = config.batchSize;
    const numBatch = Math.floor(dataHandlerOnset.getDataSize() / batchSize);
    const epochs = numEpochs;
    const testBatchSize = config.testBatchSize;
    let optimizer = config.optimizer;
    const logMessage = console.log;

    const originalDim = this.modelConfig.originalDim;

    // Reset training state
    bestValLoss = Infinity;
    epochsWithoutImprovement = 0;
    currentLearningRate = LEARNING_RATE;

    Max.outlet("training", 1);
    for (let i = 0; i < epochs; i++) {
      if (this.shouldStopTraining) break;

      let batchInputOn, batchInputVel, batchInputDur, batchInputTime;
      let testBatchInputOn, testBatchInputVel, testBatchInputDur, testBatchInputTime;
      let trainLoss;
      let epochLoss, valLoss;

      logMessage(`[Epoch ${i + 1}]\n`);
      Max.outlet("epoch", i + 1, epochs);
      
      // Calculate current weight schedules for logging
      const current_kl_weight = Math.min(1.0, i / KL_ANNEALING_EPOCHS);
      const current_timeshift_weight = Math.min(1.0, (i + 1) / TIMESHIFT_RAMP_EPOCHS);
      
      utils.log_status(`Epoch: ${i + 1} (LR: ${currentLearningRate.toFixed(6)}, KL: ${current_kl_weight.toFixed(3)}, TS: ${current_timeshift_weight.toFixed(3)})`);

      epochLoss = 0;
      
      // Learning rate decay - less aggressive schedule
      if (i > 0 && i % LEARNING_RATE_DECAY_EPOCHS === 0) {
        currentLearningRate *= LEARNING_RATE_DECAY;
        optimizer = tf.train.adam(currentLearningRate);
        logMessage(`\tLearning rate decayed to: ${currentLearningRate.toFixed(6)}`);
      }

      // Training 
      for (let j = 0; j < numBatch; j++) {
        batchInputOn = dataHandlerOnset.nextTrainBatch(batchSize).xs.reshape([batchSize, originalDim]);
        batchInputVel = dataHandlerVelocity.nextTrainBatch(batchSize).xs.reshape([batchSize, originalDim]);
        batchInputDur = dataHandlerDuration.nextTrainBatch(batchSize).xs.reshape([batchSize, originalDim]);
        batchInputTime = dataHandlerTimeshift.nextTrainBatch(batchSize).xs.reshape([batchSize, originalDim]);
        
        // Training step with gradient clipping
        trainLoss = await optimizer.minimize(() => {
          return this.vaeLoss([batchInputOn, batchInputVel, batchInputDur, batchInputTime], 
              this.apply([batchInputOn, batchInputVel, batchInputDur, batchInputTime]), i);
        }, true);
        
        trainLoss = Number(trainLoss.dataSync());
        epochLoss = epochLoss + trainLoss;

        await tf.nextFrame();
      }
      epochLoss = epochLoss / numBatch; // average
      logMessage(`\t[Average] Training Loss: ${epochLoss.toFixed(3)}. Epoch ${i} / ${epochs} \n`);
      Max.outlet("loss", epochLoss);

      // Validation 
      testBatchInputOn = dataHandlerOnset.nextTestBatch(testBatchSize).xs.reshape([testBatchSize, originalDim]);
      testBatchInputVel = dataHandlerVelocity.nextTestBatch(testBatchSize).xs.reshape([testBatchSize, originalDim]);
      testBatchInputDur = dataHandlerDuration.nextTestBatch(testBatchSize).xs.reshape([testBatchSize, originalDim]);
      testBatchInputTime = dataHandlerTimeshift.nextTestBatch(testBatchSize).xs.reshape([testBatchSize, originalDim]);
      valLoss = this.vaeLoss([testBatchInputOn, testBatchInputVel, testBatchInputDur, testBatchInputTime], 
                                this.apply([testBatchInputOn, testBatchInputVel, testBatchInputDur, testBatchInputTime]), i);
      valLoss = Number(valLoss.dataSync());

      logMessage(`\tVal Loss: ${valLoss.toFixed(3)}. Epoch ${i} / ${epochs}\n`);
      Max.outlet("val_loss", valLoss);

      // Monitor timeshift output variance to detect collapse
      const [, , , decodedTimeshift] = this.apply([testBatchInputOn, testBatchInputVel, testBatchInputDur, testBatchInputTime]);
      const timeshiftVariance = tf.moments(decodedTimeshift).variance.dataSync()[0];
      logMessage(`\tTimeshift variance: ${timeshiftVariance.toFixed(6)} (target > 0.01)`);
      
      // Warn if timeshift variance is too low (indicating collapse)
      if (timeshiftVariance < 0.01 && i > KL_ANNEALING_EPOCHS) {
        logMessage(`\t⚠️  WARNING: Timeshift variance low - potential feature collapse`);
      }

      // Early stopping check
      if (valLoss < bestValLoss) {
        bestValLoss = valLoss;
        epochsWithoutImprovement = 0;
        logMessage(`\tNew best validation loss: ${bestValLoss.toFixed(3)}`);
      } else {
        epochsWithoutImprovement++;
        if (epochsWithoutImprovement >= EARLY_STOPPING_PATIENCE) {
          logMessage(`\tEarly stopping: no improvement for ${EARLY_STOPPING_PATIENCE} epochs`);
          utils.log_status(`Early stopping at epoch ${i + 1}`);
          break;
        }
      }

      await tf.nextFrame();
    }
    this.isTrained = true;
    this.isTraining = false;
    Max.outlet("training", 0);
    utils.log_status("Training finished!");
  }
  
  generate(zs){
    let [outputsOn, outputsVel, outputsDur, outputsTime] = this.decoder.apply(zs);

    outputsOn = outputsOn.reshape([NUM_MIDI_CLASSES, LOOP_DURATION]);     
    outputsVel = outputsVel.reshape([NUM_MIDI_CLASSES, LOOP_DURATION]);    
    outputsDur = outputsDur.reshape([NUM_MIDI_CLASSES, LOOP_DURATION]);    
    outputsTime = outputsTime.reshape([NUM_MIDI_CLASSES, LOOP_DURATION]);    

    return [outputsOn.arraySync(), outputsVel.arraySync(), outputsDur.arraySync(), outputsTime.arraySync()];
  }

  bendModel(noise_range){
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

  async saveModel(path){
    const saved = await this.decoder.save(path);
    utils.post(saved);
  }

  async loadModel(path){
    this.decoder = await tf.loadLayersModel(path);
    this.isTrained = true;
  }

  encode(inputOn, inputVel, inputDur, inputTime){
    if (!this.encoder) {
      utils.error_status("Model is not trained yet");
      return;
    }

    // reshaping...
    inputOn = inputOn.reshape([1, ORIGINAL_DIM]);
    inputVel = inputVel.reshape([1, ORIGINAL_DIM]);
    inputDur = inputDur.reshape([1, ORIGINAL_DIM]);
    inputTime = inputTime.reshape([1, ORIGINAL_DIM]);
    
    let [zMean, zLogVar, zs] = this.encoder.apply([inputOn, inputVel, inputDur, inputTime]);
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

