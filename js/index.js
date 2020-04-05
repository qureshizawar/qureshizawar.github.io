tf.setBackend('webgl');

let model_classifier;

let model_depth_encoder;
let model_depth_decoder;

const IMAGE_HEIGHT = 224;
const IMAGE_WIDTH = 384;

const Depth_IMAGE_HEIGHT = 192;
const Depth_IMAGE_WIDTH = 640;

var cors_api_url = 'https://cors-anywhere.herokuapp.com/';

const status_classifier = document.getElementById('status_classifier');
const status_depth = document.getElementById('status_depth');

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.getElementById('inpimg');
      img.src = e.target.result;
      img.height = IMAGE_HEIGHT;
      img.width = IMAGE_WIDTH;
      img.onload = () => Depth_Demo(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

document.getElementById('btn').onclick = function() {
    let url = new URL(document.getElementById('imagename').value);
    //console.log(url)

    var request = new XMLHttpRequest();
    request.open('GET', cors_api_url + url, true);
    request.responseType = 'blob';
    request.send();

    request.onload = function() {
        var reader = new FileReader();
        reader.readAsDataURL(request.response);
        reader.onload = e => {
            //console.log('DataURL:', e.target.result);
          // Fill the image & call predict.
          let img = document.getElementById('inpimg');
          img.src = e.target.result;
          img.height = IMAGE_HEIGHT;
          img.width = IMAGE_WIDTH;
          img.onload = () => Depth_Demo(img);
        };
    };
}

const filesElement0 = document.getElementById('files0');
filesElement0.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.getElementById('inpimg0');
      img.src = e.target.result;
      img.height = IMAGE_HEIGHT;
      img.width = IMAGE_HEIGHT;
      img.onload = () => classifier_Demo(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

document.getElementById('btn0').onclick = function() {
    //var val = document.getElementById('imagename0').value;
    let url = new URL(document.getElementById('imagename0').value);

    var request = new XMLHttpRequest();
    request.open('GET', cors_api_url + url, true);
    request.responseType = 'blob';
    request.send();

    request.onload = function() {
        var reader = new FileReader();
        reader.readAsDataURL(request.response);
        reader.onload = e => {
            //console.log('DataURL:', e.target.result);
          // Fill the image & call predict.
          let img = document.getElementById('inpimg0');
          img.src = e.target.result;
          img.height = IMAGE_HEIGHT;
          img.width = IMAGE_HEIGHT;
          img.onload = () => classifier_Demo(img);
        };
    };
}


const ClassiferWarmup = async () => {

  status_classifier.textContent = 'Status: Loading...';
  model_classifier = await tf.loadGraphModel('/assets/tfjs_model/model.json');

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  //model_classifier.predict(tf.zeros([1, IMAGE_HEIGHT, IMAGE_HEIGHT, 3])).dispose();

  // Make a prediction through the locally hosted inpimg0.jpg.
  const inpElement = document.getElementById('inpimg0');
  if (inpElement.complete && inpElement.naturalHeight !== 0) {
    classifier_Demo(inpElement);
    inpElement.style.display = '';
  } else {
    inpElement.onload = () => {
      classifier_Demo(inpElement);
      inpElement.style.display = '';
    }
  }

  document.getElementById('file-container').style.display = '';
};


const classifier_Demo = async (imElement) => {
    // const imElement = document.getElementById('inpimg');
    status_classifier.textContent = 'Status: Loading...';
    // The first start time includes the time it takes to extract the image
    // from the HTML and preprocess it, in additon to the predict() call.
    //const startTime1 = performance.now();
    const predictions = tf.tidy(() => {

    const img = tf.browser.fromPixels(imElement).toFloat();
    const scale = tf.scalar(255.);
    const mean = tf.tensor3d([0.485, 0.456, 0.406], [1,1,3]);
    const std = tf.tensor3d([0.229, 0.224, 0.225], [1,1,3]);
    const normalised = img.div(scale).sub(mean).div(std);
    //const model = await tf.loadGraphModel('/assets/tfjs_model/model.json');
    status_classifier.textContent = 'Status: Model loaded! running inference';
    //const batched = normalised.transpose([2,0,1]).expandDims();
    const batched = normalised.transpose([0,1,2]).expandDims();

    //const predictions = model_classifier.predict(batched);
    //startTime2 = performance.now();
    return  model_classifier.predict(batched);

    });

    var output = [];

    out = predictions.arraySync();
    output.push(["bus", out[0][0]]);
    output.push(["car", out[0][1]]);
    output.push(["pickup", out[0][2]]);
    output.push(["truck", out[0][3]]);
    output.push(["van", out[0][4]]);

    //console.log(output);
    // done to sort vals as numbers instead of strings
    output.sort(function(a, b){return b[1] - a[1]});
    document.getElementById("classifier_out1").innerHTML = output[0][0]+": "+(output[0][1]*100).toFixed(2)+"%";
    document.getElementById("classifier_out2").innerHTML = output[1][0]+": "+(output[1][1]*100).toFixed(2)+"%";
    document.getElementById("classifier_out3").innerHTML = output[2][0]+": "+(output[2][1]*100).toFixed(2)+"%";

    /*const totalTime1 = performance.now() - startTime1;
    const totalTime2 = performance.now() - startTime2;
    status_classifier.textContent =  `Done in ${Math.floor(totalTime1)} ms ` +
        `(not including preprocessing: ${Math.floor(totalTime2)} ms)`;*/

    status_classifier.textContent = 'Status: Done!';

    //console.log("before: ", tf.memory());
    predictions.dispose();
    //tf.disposeVariables();
    //console.log("after: ", tf.memory());
  };


  const DepthWarmup = async () => {

    status_depth.textContent = 'Status: Loading...';
    model_depth_encoder = await tf.loadLayersModel('/assets/tfjs_encoder_quant/model.json');
    model_depth_decoder = await tf.loadLayersModel('/assets/tfjs_decoder_quant/model.json');

    // Warmup the model. This isn't necessary, but makes the first prediction
    // faster. Call `dispose` to release the WebGL memory allocated for the return
    // value of `predict`.
    // model.predict(tf.zeros([1, 3, IMAGE_HEIGHT, IMAGE_WIDTH])).dispose();

    // Make a prediction through the locally hosted inpimg.jpg.
    let inpElement = document.getElementById('inpimg');
    //inpElement.src = e.target.result;
    if (inpElement.complete && inpElement.naturalHeight !== 0) {
      Depth_Demo(inpElement);
      inpElement.style.display = '';
    } else {
      inpElement.onload = () => {
        Depth_Demo(inpElement);
        inpElement.style.display = '';
      }
    }

    document.getElementById('file-container').style.display = '';
  };


  const Depth_Demo = async (imElement) => {
      // triplet: depth, normals, segmentation
    // const imElement = document.getElementById('inpimg');

    status_depth.textContent = 'Status: Loading...';
    // The first start time includes the time it takes to extract the image
    // from the HTML and preprocess it, in additon to the predict() call.
    //const startTime1 = performance.now();
    const depthMask = tf.tidy(() => {

    var img = tf.image.resizeBilinear(tf.browser.fromPixels(imElement).toFloat(),
      [Depth_IMAGE_HEIGHT,Depth_IMAGE_WIDTH]);
    //console.log(img);
    const scale = tf.scalar(255.);
    //const mean = tf.tensor3d([0.485, 0.456, 0.406], [1,1,3]);
    //const std = tf.tensor3d([0.229, 0.224, 0.225], [1,1,3]);
    //const normscale = tf.tensor3d([1., 1., -1.], [1,1,3]);
    //const normsub = tf.tensor3d([-1., -1., 1], [1,1,3]);
    img = img.div(scale);//.sub(mean).div(std);

    status_depth.textContent = 'Status: Model loaded! running inference';
    img = img.transpose([2,0,1]).expandDims();

    //const features = model_depth_encoder.predict(batched);
    const predictions = model_depth_decoder.predict(model_depth_encoder.predict(img));

    //console.log(predictions);

    const depthPred = predictions[3].squeeze(0).transpose([1,2,0]);
    //const MAX_D = depthPred.max();
    //const MIN_D = depthPred.min();

    //const depthMask = depthPred.sub(depthPred.min()).divNoNan(depthPred.max().sub(depthPred.min()));
    //startTime2 = performance.now();
    return depthPred.sub(depthPred.min()).divNoNan(depthPred.max().sub(depthPred.min()));

    });

    const depthCanvas = document.getElementById('depth');

    await tf.browser.toPixels(tf.image.resizeBilinear(depthMask,
      [IMAGE_HEIGHT,IMAGE_WIDTH]), depthCanvas);

    /*const totalTime1 = performance.now() - startTime1;
    const totalTime2 = performance.now() - startTime2;
    status_depth.textContent =  `Done in ${Math.floor(totalTime1)} ms ` +
        `(not including preprocessing: ${Math.floor(totalTime2)} ms)`;*/

    status_depth.textContent = 'Status: Done!';
    //console.log("before: ", tf.memory());

    depthMask.dispose();
    //tf.disposeVariables();
    //console.log("after: ", tf.memory());
  };

  ClassiferWarmup();
  DepthWarmup();
