tf.setBackend('webgl');
// Even though it is an FCN-model
// smaller image size is preferrable for the demonstration purposes
// You can set any image size that even divides 32
const IMAGE_HEIGHT = 224;
const IMAGE_WIDTH = 384;

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
  const model = await tf.loadGraphModel('/assets/tfjs_model/model.json');

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  //model.predict(tf.zeros([1, IMAGE_HEIGHT, IMAGE_HEIGHT, 3])).dispose();

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
    const offset = 0; // label mask offset
    const img = tf.browser.fromPixels(imElement).toFloat();
    const scale = tf.scalar(255.);
    const mean = tf.tensor3d([0.485, 0.456, 0.406], [1,1,3]);
    const std = tf.tensor3d([0.229, 0.224, 0.225], [1,1,3]);
    const normalised = img.div(scale).sub(mean).div(std);
    const model = await tf.loadGraphModel('/assets/tfjs_model/model.json');
    status_classifier.textContent = 'Status: Model loaded! running inference';
    //const batched = normalised.transpose([2,0,1]).expandDims();
    const batched = normalised.transpose([0,1,2]).expandDims();

    const predictions = model.predict(batched);

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
    document.getElementById("classifier_out1").innerHTML = output[0];
    document.getElementById("classifier_out2").innerHTML = output[1];
    document.getElementById("classifier_out3").innerHTML = output[2];

    predictions.dispose()
    status_classifier.textContent = 'Status: Done!';
  };


  const DepthWarmup = async () => {

    status_depth.textContent = 'Status: Loading...';
    const model = await tf.loadLayersModel('/assets/tfjs_depth_quant/model.json');

    // Warmup the model. This isn't necessary, but makes the first prediction
    // faster. Call `dispose` to release the WebGL memory allocated for the return
    // value of `predict`.
    // model.predict(tf.zeros([1, 3, IMAGE_HEIGHT, IMAGE_WIDTH])).dispose();

    // Make a prediction through the locally hosted inpimg.jpg.
    const inpElement = document.getElementById('inpimg');
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
    const offset = 1; // label mask offset
    const img = tf.browser.fromPixels(imElement).toFloat();
    //console.log(img);
    const scale = tf.scalar(255.);
    //const mean = tf.tensor3d([0.485, 0.456, 0.406], [1,1,3]);
    //const std = tf.tensor3d([0.229, 0.224, 0.225], [1,1,3]);
    const normscale = tf.tensor3d([1., 1., -1.], [1,1,3]);
    const normsub = tf.tensor3d([-1., -1., 1], [1,1,3]);
    const normalised = img.div(scale)//.sub(mean).div(std);
    const model = await tf.loadLayersModel('/assets/tfjs_depth_quant/model.json');
    status_depth.textContent = 'Status: Model loaded! running inference';
    const batched = normalised.transpose([2,0,1]).expandDims();

    const predictions = model.predict(batched);


    //console.log(predictions);
    const initShape = batched.shape.slice(2,4);

    const depthPred = predictions.squeeze(0).transpose([1,2,0]);
    const MAX_D = depthPred.max();
    const MIN_D = depthPred.min();

    const depthMask = depthPred.sub(MIN_D).divNoNan(MAX_D.sub(MIN_D));

    const depthCanvas = document.getElementById('depth');

    predictions.dispose()
    depthPred.dispose()

    status_depth.textContent = 'Status: Done!';
    await tf.browser.toPixels(depthMask, depthCanvas);
  };

  ClassiferWarmup();
  DepthWarmup();
