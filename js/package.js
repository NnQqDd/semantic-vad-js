export function makeCallable(instance) {
  const fn = (...x) => instance.call(...x);

  return new Proxy(fn, {
    get(_, prop) {
      return instance[prop];
    },
    set(_, prop, value) {
      instance[prop] = value;
      return true;
    }
  });
}


export function isTfTensor(x) {
  return (
    x != null &&
    typeof x === 'object' &&
    Array.isArray(x.shape) &&      // shape exists like TF.js
    typeof x.data === 'function' && // has async data()
    typeof x.dataSync === 'function' && // has sync accessor
    typeof x.dtype === 'string'    // has dtype string
  );
}


/**
 * Map a TF.js dtype string to its corresponding ORT dtype string + TypedArray constructor.
 * @param {string} tfDtype
 * @returns {{ ortDtype: string, TypedArray: any }}
 */
function resolveDtype(tfDtype) {
  const map = {
    float32: { ortDtype: 'float32',  TypedArray: Float32Array   },
    float64: { ortDtype: 'double',   TypedArray: Float64Array   },
    int32:   { ortDtype: 'int32',    TypedArray: Int32Array     },
    uint8:   { ortDtype: 'uint8',    TypedArray: Uint8Array     },
    uint16:  { ortDtype: 'uint16',   TypedArray: Uint16Array    },
    int16:   { ortDtype: 'int16',    TypedArray: Int16Array     },
  };

  const resolved = map[tfDtype];
  if (!resolved) throw new Error(`Unsupported TF.js dtype: ${tfDtype}`);
  return resolved;
}

/**
 * Convert a TF.js tensor to an ort.Tensor.
 * @param {tf.Tensor} tfTensor
 * @returns {Promise<ort.Tensor>}
 */
export function tfTensorToOrt(tfTensor) {
  const { ortDtype, TypedArray } = resolveDtype(tfTensor.dtype);
  const shape = tfTensor.shape;
  const data = tfTensor.dataSync();
  return new ort.Tensor(ortDtype, data, shape);
}


/**
 * Convert an ort.Tensor to a TF.js tensor.
 * @param {ort.Tensor} ortTensor
 * @returns {tf.Tensor}
 */
export function ortTensorToTf(ortTensor) {
  const { type: ortDtype, dims, data } = ortTensor;
  const dtypeMap = {
    float32: 'float32',
    double:  'float64',
    int32:   'int32',
    uint8:   'uint8',
    uint16:  'uint16',
    int16:   'int16',
  };

  const tfDtype = dtypeMap[ortDtype];
  if (!tfDtype) throw new Error(`Unsupported ORT dtype: ${ortDtype}`);
  const shape = Array.from(dims);
  return tf.tensor(data, shape, tfDtype);
}


export class PretrainedONNXModel {
  constructor(deviceId=null, inputNames=null, outputNames=null) {
    this.inputNames = inputNames;
    this.outputNames = outputNames;
    this.loaded = false;
    this.deviceId = deviceId;
    this.session = null;
    this.pretrainedModelPath = null;
  }

  static async fromFile(pretrainedModelPath, deviceId=null, inputNames=null, outputNames=null, load=false) {
    const instance = new PretrainedONNXModel(deviceId, inputNames, outputNames);
    instance.pretrainedModelPath = pretrainedModelPath;
    if (load) {
      await instance.load();
    }
    return instance;
  }

  get device() {
    return this.deviceId === null ? "cpu" : `cuda:${this.deviceId}`;
  }

  unload() {
    this.loaded = false;
    if (this.session) {
      // onnxruntime sessions typically don't have an explicit close; drop reference
      this.session = null;
    }
  }

  async load() {
    if (this.loaded) return;
    let ort;
    try {
      ort = require("onnxruntime-node");
    } catch (e) {
      // fallback to global ort (e.g., onnxruntime-web loaded via script tag)
      ort = globalThis.ort;
    }

    if (!ort || !ort.InferenceSession) {
      throw new Error(
        "onnxruntime not found. Install `onnxruntime-node` (Node) or load `onnxruntime-web` (browser)."
      );
    }

    const createOptions = {};

    if (this.deviceId === null) {
      createOptions.executionProviders = ["cpu"];
    } else {
      createOptions.executionProviders = ["cuda", "cpu"];
      createOptions.deviceId = this.deviceId;
    }

    try {
      this.session = await ort.InferenceSession.create(
        this.pretrainedModelPath,
        createOptions
      );
      this.loaded = true;
    } catch (err) {
      try {
        const altOpts = { providers: createOptions.executionProviders };
        this.session = await ort.InferenceSession.create(
          this.pretrainedModelPath,
          altOpts
        );
        this.loaded = true;
      } catch (err2) {
        const message = `Failed to create ONNX InferenceSession: ${err.message}; fallback error: ${err2.message}`;
        throw new Error(message);
      }
    }
  }

  async run(inputs={}, options={}) {
    const retry = options.retry !== undefined ? !!options.retry : true;
    if (!this.loaded) {
      await this.load();
    }

    try {
      const output = await this.session.run(inputs);
      return output;
    } catch (err) {
      if (retry) {
        try {
          this.unload();
          await this.load();
          return await this.run(inputs, { retry: false });
        } catch (innerErr) {
          throw innerErr;
        }
      }
      throw err;
    }
  }

  async call(...inputTensors) {
    let inputs = {};
    let isTensor = false;
    for (let i = 0; i < this.inputNames.length; i++) {
      if (isTfTensor(inputTensors[i])){
        isTensor = true;
        inputTensors[i] = tfTensorToOrt(inputTensors[i]);
      }
      inputs[this.inputNames[i]] = inputTensors[i];
    }

    let output = await this.session.run(inputs);
    let outputs = []
    if(isTensor){
      for (let i = 0; i < this.outputNames.length; i++){
        outputs.push(ortTensorToTf(output[this.outputNames[i]]));
      } 
    }
    else{
      for (let i = 0; i < this.outputNames.length; i++){
        outputs.push(output[this.outputNames[i]])
      } 
    }
    if (outputs.length === 1){
      return outputs[0];
    }
    return outputs;
  }

  getInputMetadata() {
    return this.session.inputMetadata;
  }

  getOutputMetadata() {
    return this.session.outputMetadata;
  }
}


export async function loadAudio(url, sampleRate=16000){
  const res = await fetch(url);
  const buffer = await res.arrayBuffer();

  const ctx = new AudioContext({ sampleRate: sampleRate });
  const audio = await ctx.decodeAudioData(buffer);
  const channel = audio.getChannelData(0); // Float32Array

  return channel;
}


export function withMask(waveform, maxLength=128000, paddingValue=0.0){
  const realSamples = Math.min(waveform.length, maxLength);
  const padWave = new Float32Array(maxLength);
  if (paddingValue !== 0.0) {
    padWave.fill(paddingValue);
  }
  padWave.set(waveform.subarray(0, realSamples), 0);

  const attMask = new Uint8Array(maxLength);
  for (let i = 0; i < realSamples; i++) {
    attMask[i] = 1;
  }
  return [padWave, attMask];
}


export async function loadAudioWithMask(
  url,
  sampleRate=16000,
  maxLength=128000, 
  paddingValue=0.0
) {
  const channel = await loadAudio(url, sampleRate);
  return withMask(channel, maxLength, paddingValue);
}



export function zeroMeanUnitVarNorm(inputValues, attentionMasks=null, paddingValue=0.0) {
  const EPS = 1e-9;
  const out = new Array(inputValues.length);

  if (attentionMasks != null) {
    // Ensure masks is an array-like the same length as inputValues
    for (let i = 0; i < inputValues.length; ++i) {
      const vec = inputValues[i];
      const mask = attentionMasks[i];

      // compute length = sum(mask)
      let length = 0;
      if (mask && mask.length) {
        for (let j = 0; j < mask.length; ++j) {
          // coerce mask value to 0/1 in case it's not typed
          length += mask[j] ? 1 : 0;
        }
      }

      const n = vec.length;
      // If length is 0, produce padded output (all paddingValue)
      if (length == 0) {
        const padded = new Float32Array(n);
        if (paddingValue !== 0.0) {
          for (let k = 0; k < n; ++k) padded[k] = paddingValue;
        }
        out[i] = padded;
        continue;
      }

      // compute mean over vec[0:length]
      let sum = 0.0;
      for (let k = 0; k < length; ++k) sum += vec[k];
      const mean = sum / length;

      // compute variance over vec[0:length]
      let sq = 0.0;
      for (let k = 0; k < length; ++k) {
        const d = vec[k] - mean;
        sq += d * d;
      }
      const variance = sq / length;
      const std = Math.sqrt(variance + EPS);

      // allocate result and normalize first `length` samples
      const normed = new Float32Array(n);
      for (let k = 0; k < length; ++k) {
        normed[k] = (vec[k] - mean) / std;
      }

      // set remainder to paddingValue (if any)
      if (length < n) {
        if (paddingValue === 0.0) {
          // already zeros, nothing to do
        } else {
          for (let k = length; k < n; ++k) normed[k] = paddingValue;
        }
      }

      out[i] = normed;
    }
  } 

  else {
    // No attention mask: normalize whole vector
    for (let i = 0; i < inputValues.length; ++i) {
      const vec = inputValues[i];
      const n = vec.length;

      // compute mean
      let sum = 0.0;
      for (let k = 0; k < n; ++k) sum += vec[k];
      const mean = sum / n;

      // compute variance
      let sq = 0.0;
      for (let k = 0; k < n; ++k) {
        const d = vec[k] - mean;
        sq += d * d;
      }
      const variance = sq / n;
      const std = Math.sqrt(variance + EPS);
      const normed = new Float32Array(n);
      for (let k = 0; k < n; ++k) normed[k] = (vec[k] - mean) / std;

      out[i] = normed;
    }
  }

  return out;
}


export class SileroVADStream{
  constructor(){}
  
  static async fromFile(sileroVADONNXPath) {
    let model = await PretrainedONNXModel.fromFile(sileroVADONNXPath, null, ["input", "state", "sr"], ["output", "stateN"], true);
    console.log(model.constructor.name, model.getInputMetadata());
    console.log(model.constructor.name, model.getOutputMetadata());
    
    let instance = new SileroVADStream();
    instance.model = makeCallable(model);
    instance.state = new ort.Tensor(
      "float32",
      new Float32Array(2 * 1 * 128),
      [2, 1, 128]
    );
    instance.sr = new ort.Tensor("int64", new BigInt64Array([16000n]), []);
    
    return makeCallable(instance);
  }

  async call(samples){ // Float32Array 
    if(!(samples instanceof Float32Array)){
      throw new Error(`Samples are not of Float32Array instance!`);
    }
    if(samples.length === 0){
      throw new Error(`Empty waveform!`);
    }
    if(samples.length % 512 != 0){
      throw new Error(`Samples length must be a multiplier of 512`)
    }
    let ci = 0;
    let results = [];
    for(ci = 0; ci < samples.length; ci += 512){
      let end = Math.min(ci + 512, samples.length);
      let chunk = samples.subarray(ci, end);
      let input = new ort.Tensor("float32", chunk, [1, 512]);
      let out = await this.model(input, this.state, this.sr);
      this.state = out[1];
      results.push(out[0].data[0]);
    }
    return results;
  }

  reset(){
    this.state = new ort.Tensor(
      "float32",
      new Float32Array(2 * 1 * 128),
      [2, 1, 128]
    );
  }
}


import { AutoProcessor } from "https://cdn.jsdelivr.net/npm/@xenova/transformers/dist/transformers.min.js";
export { AutoProcessor };
export class SmartTurnV3{
  constructor(){}

  static async fromFile(smartTurnONNXPath, configDir){
    let model = await PretrainedONNXModel.fromFile(smartTurnONNXPath, null, ["input_features"], ["logits"], true);
    console.log(model.constructor.name, model.getInputMetadata());
    console.log(model.constructor.name, model.getOutputMetadata());

    let config = await (await fetch(configDir + '/preprocessor_config.json')).json();

    let instance = new SmartTurnV3();
    instance.maxLength = config.n_samples;
    instance.model = makeCallable(model);
    instance.preprocessor = await AutoProcessor.from_pretrained(configDir);
    instance.shapes = [1, config.feature_size, config.nb_max_frames]
    return makeCallable(instance);
  }

  static lastSamplesPadLeft(waveform, targetLength=128000) {
    const out = new Float32Array(targetLength);

    if (waveform.length >= targetLength) {
      // take last samples
      out.set(waveform.subarray(waveform.length - targetLength));
    } else {
      // left pad with zeros
      const offset = targetLength - waveform.length;
      out.set(waveform, offset);
    }

    return out;
  }

  // static sumMeanStd(inputValues) {
  //   const EPS = 1e-9;

  //   const vec = inputValues;
  //   const n = vec.length;

  //   let sum = 0.0;
  //   for (let k = 0; k < n; ++k) sum += vec[k];
  //   const mean = sum / n;

  //   let sq = 0.0;
  //   for (let k = 0; k < n; ++k) {
  //     const d = vec[k] - mean;
  //     sq += d * d;
  //   }

  //   const variance = sq / n;
  //   const std = Math.sqrt(variance + EPS);

  //   return [sum, mean, std];
  // }

  async call(waveform){
    if(!(waveform instanceof Float32Array)){
      throw new Error(`Samples are not of Float32Array instance!`);
    }
    if(waveform.length === 0){
      return 0.0;
    }
    
    waveform = SmartTurnV3.lastSamplesPadLeft(waveform, this.maxLength);
    waveform = zeroMeanUnitVarNorm([waveform])[0];

    let {input_features, _} = await this.preprocessor(waveform);
    let spec = new ort.Tensor(
      "float32",
      input_features.data,
      this.shapes
    );
    let out = await this.model(spec);

    return out.data[0];
  }


}