import * as tf from '@tensorflow/tfjs';

class sampleLayer extends tf.layers.Layer {
    constructor(args) {
      super({});
    }
  
    computeOutputShape(inputShape) {
      return inputShape[0];
    }
  
    call(inputs, kwargs) {
      return tf.tidy(() => {
        const [z_mean, z_log_var] = inputs;
        const batch = z_mean.shape[0];
        const dim = z_mean.shape[1];
        const epsilon = tf.randomNormal([batch, dim]);
        const half = tf.scalar(0.5);
        const temp = z_log_var.mul(half).exp().mul(epsilon);
        const sample = z_mean.add(temp);
        return sample;
      });
    }
  
    getClassName() {
      return 'sampleLayer';
    }
  }
  
  async function train(config) {
    // # network parameters
    const input_shape = config.input_shape;
    const original_dim = config.original_dim;
    const intermediate_dim = config.intermediate_dim;
    const batch_size = config.batch_size;
    const num_batch = config.num_batch;
    const latent_dim = config.latent_dim;
    const epochs = config.epochs;
    const test_batch_size = 1000;
  
    // # build encoder model
    const encoder_inputs = tf.input({shape: [original_dim]});
    const x1_l = tf.layers.dense({units: intermediate_dim, kernelInitializer: 'glorotUniform'}).apply(encoder_inputs);
    const x1_n = tf.layers.batchNormalization({axis:1}).apply(x1_l);
    const x1 = tf.layers.elu().apply(x1_n);
    const z_mean = tf.layers.dense({units: latent_dim, kernelInitializer: 'glorotUniform'}).apply(x1);
    const z_log_var = tf.layers.dense({units: latent_dim, kernelInitializer: 'glorotUniform'}).apply(x1);
  
  
    // # use reparameterization trick to push the sampling out as input
    // # note that "output_shape" isn't necessary with the TensorFlow backend
    const z = new sampleLayer().apply([z_mean, z_log_var]);
    const encoder = tf.model({
      inputs: encoder_inputs,
      outputs: [
        z_mean, z_log_var, z
      ],
      name: "encoder"
    })
  
    // # build decoder model
    const decoder_inputs = tf.input({shape: [latent_dim]});
    const x2_l = tf.layers.dense({units: intermediate_dim, kernelInitializer: 'glorotUniform'}).apply(decoder_inputs);
    const x2_n = tf.layers.batchNormalization({axis: 1}).apply(x2_l);
    const x2 = tf.layers.elu().apply(x2_n);
    const decoder_outputs = tf.layers.dense({units: original_dim, activation: 'sigmoid'}).apply(x2);
    const decoder = tf.model({inputs: decoder_inputs, outputs: decoder_outputs, name: "decoder"})
  
    const vae = (inputs) => {
      return tf.tidy(() => { //avoid memory leakage
        const [z_mean, z_log_var, z] = encoder.apply(inputs);
        const outputs = decoder.apply(z);
        return [z_mean, z_log_var, outputs];
      })
    }
  
    const optimizer = tf.train.adam();
  
    const reconstructionLoss = (yTrue, yPred) => {
      return tf.tidy(() => { //avoid memory leakage
        let reconstruction_loss;
        reconstruction_loss = tf.metrics.binaryCrossentropy(yTrue, yPred)
        reconstruction_loss = reconstruction_loss.mul(tf.scalar(yPred.shape[1]));
        return reconstruction_loss;
      });
    }
  
    const klLoss = (z_mean, z_log_var) => {
      return tf.tidy(() => { //avoid memory leakage
        let kl_loss;
        kl_loss = tf.scalar(1).add(z_log_var).sub(z_mean.square()).sub(z_log_var.exp());
        kl_loss = tf.sum(kl_loss, -1);
        kl_loss = kl_loss.mul(tf.scalar(-0.5));
        return kl_loss;
      });
    }
  
    const vaeLoss = (yTrue, yPred) => {
      return tf.tidy(() => { //avoid memory leakage
        const [z_mean, z_log_var, y] = yPred;
        const reconstruction_loss = reconstructionLoss(yTrue, y);
        const kl_loss = klLoss(z_mean, z_log_var);
        const total_loss = tf.mean(reconstruction_loss.add(kl_loss));
        return total_loss;
      });
    }
