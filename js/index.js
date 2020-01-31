tf.setBackend('webgl');
// Even though it is an FCN-model
// smaller image size is preferrable for the demonstration purposes
// You can set any image size that even divides 32
const IMAGE_HEIGHT = 224;
const IMAGE_WIDTH = 384;

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


const classifier_Demo = async (imElement) => {
    // const imElement = document.getElementById('inpimg');
    status_classifier.textContent = 'Loading model...';
    const offset = 0; // label mask offset
    const img = tf.browser.fromPixels(imElement).toFloat();
    const scale = tf.scalar(255.);
    const mean = tf.tensor3d([0.485, 0.456, 0.406], [1,1,3]);
    const std = tf.tensor3d([0.229, 0.224, 0.225], [1,1,3]);
    const normalised = img.div(scale).sub(mean).div(std);
    const model = await tf.loadGraphModel('/assets/tfjs_model/model.json');
    status_classifier.textContent = 'Model loaded! running inference';
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
    status_classifier.textContent = 'Done!';
    document.getElementById("classifier_out").innerHTML = output[0];
  };


  const Depth_Demo = async (imElement) => {
      // triplet: depth, normals, segmentation
    // const imElement = document.getElementById('inpimg');

    status_depth.textContent = 'Loading model...';
    const offset = 1; // label mask offset
    const img = tf.browser.fromPixels(imElement).toFloat();
    //console.log(img);
    const scale = tf.scalar(255.);
    //const mean = tf.tensor3d([0.485, 0.456, 0.406], [1,1,3]);
    //const std = tf.tensor3d([0.229, 0.224, 0.225], [1,1,3]);
    const normscale = tf.tensor3d([1., 1., -1.], [1,1,3]);
    const normsub = tf.tensor3d([-1., -1., 1], [1,1,3]);
    const normalised = img.div(scale)//.sub(mean).div(std);
    const model = await tf.loadLayersModel('/assets/tfjs_depth/model.json');
    status_depth.textContent = 'Model loaded! running inference';
    const batched = normalised.transpose([2,0,1]).expandDims();

    const predictions = model.predict(batched);


    //console.log(predictions);
    const initShape = batched.shape.slice(2,4);

    const depthPred = predictions.squeeze(0).transpose([1,2,0]);
    const MAX_D = depthPred.max();
    const MIN_D = depthPred.min();

    const depthMask = depthPred.sub(MIN_D).divNoNan(MAX_D.sub(MIN_D));

    const depthCanvas = document.getElementById('depth');
    status_depth.textContent = 'Done!';
    await tf.browser.toPixels(depthMask, depthCanvas);
  };
