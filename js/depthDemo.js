tf.setBackend('webgl');
tf.ENV.set('WEBGL_CONV_IM2COL', false);
tf.ENV.set('WEBGL_PACK', false); // This needs to be done otherwise things run very slow v1.0.4
tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true)

tf.enableProdMode();

let model_depth_encoder;
let model_depth_decoder;

// H x W
const input_res_lookup = {
  outdoors1_low: [128, 320],
  outdoors1_mid: [192, 320],
  outdoors1_high: [192, 640],
  outdoors2_low: [160, 320],
  outdoors2_mid: [256, 480],
  outdoors2_high: [320, 640],
  indoors_low: [224, 320],
  indoors_mid: [320, 448],
  indoors_high: [416, 544]
};

const model_lookup = {
  outdoors1: ['/assets/tfjs_encoder_quant/model.json',
    '/assets/tfjs_decoder_quant/model.json'
  ],
  outdoors2: ['/assets/outdoors_tfjs_encoder/model.json',
    '/assets/outdoors_tfjs_decoder/model.json'
  ],
  indoors: ['/assets/indoors_tfjs_encoder/model.json',
    '/assets/indoors_tfjs_decoder/model.json'
  ]
};

const norm_lookup = {
  outdoors1: [
    [0, 0, 0],
    [1, 1, 1]
  ],
  outdoors2: [
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
  ],
  indoors: [
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
  ]
}

const out_idx = {
  outdoors1: 3,
  outdoors2: 0,
  indoors: 0
}

var curr_model = "outdoors2"
var curr_res = "mid"

var Depth_IMAGE_HEIGHT = 192;
var Depth_IMAGE_WIDTH = 320;

var output_HEIGHT = 300;
var output_WIDTH = 400;

const status_depth = document.getElementById('status_depth');

const dropdown_depth_qual = document.getElementById('dropdown_depth_qual');
const depth_low = document.getElementById('depth_low');
const depth_medium = document.getElementById('depth_medium');
const depth_high = document.getElementById('depth_high');

if (!(is_touch_device()) && !(window.matchMedia('(max-device-width: 960px)').matches)) {
  curr_res = "high"
  dropdown_depth_qual.textContent = depth_high.textContent;
}

document.getElementById("files").addEventListener("change", function(evt) {
  file_infer(evt.target.files[0], document.getElementById('inpimg'),
    status_depth, Depth_Demo);
});
document.getElementById("depth_files_btn").addEventListener("click", function(evt) {
  file = document.getElementById("files").files[0];
  if (file == null) {
    status_depth.textContent = 'Status: File not found';
  } else {
    file_infer(file, document.getElementById('inpimg'),
      status_depth, Depth_Demo);
  }
});
document.getElementById("select_files_btn").onclick = function() {
  // event.preventDefault();
  // console.log("btn pressed!")
  status_depth.textContent = 'Status: Fetching image...';
  img_in = document.getElementById('inpimg')
  Depth_Demo(img_in);
}
document.getElementById("depth_files_btn").addEventListener("click", function(evt) {
  file = document.getElementById("files").files[0];
  if (file == null) {
    status_depth.textContent = 'Status: File not found';
  } else {
    file_infer(file, document.getElementById('inpimg'),
      status_depth, Depth_Demo);
  }
});

document.getElementById("btn").addEventListener("click", function(evt) {
  url_infer(document.getElementById('imagename'), document.getElementById('inpimg'),
    status_depth, Depth_Demo);
});

function custom_mgrid(rows, cols) {
  a = tf.tile(tf.range(0, rows).reshape([rows, 1]), [1, cols]); // cols
  b = tf.tile(tf.range(0, cols).expandDims(0), [rows, 1]); // rows
  return [a, b];
}

const update_model = async (name) => {
  // console.log("update_model!")
  // console.log(name)
  // console.log(model_lookup[name][0])
  model_depth_encoder = await tf.loadLayersModel(model_lookup[name][0]);
  model_depth_decoder = await tf.loadLayersModel(model_lookup[name][1]);

};

const DepthWarmup = async () => {

  status_depth.textContent = 'Status: Loading...';
  // model_depth_encoder = await tf.loadLayersModel('/assets/tfjs_encoder_quant/model.json');
  // model_depth_decoder = await tf.loadLayersModel('/assets/tfjs_decoder_quant/model.json');
  model_depth_encoder = await tf.loadLayersModel(model_lookup[curr_model][0]);
  model_depth_decoder = await tf.loadLayersModel(model_lookup[curr_model][1]);

  // Make a prediction through the locally hosted inpimg.jpg.
  let inpElement = document.getElementById('inpimg');
  //inpElement.src = e.target.result;
  if (inpElement.complete && inpElement.naturalHeight !== 0) {
    set_static_output_size(inpElement);
    Depth_Demo(inpElement);
    inpElement.style.display = '';
  } else {
    inpElement.onload = () => {
      set_static_output_size(inpElement);
      Depth_Demo(inpElement);
      inpElement.style.display = '';
    }
  }

  // document.getElementById('file-container').style.display = '';
};

const Depth_Demo = async (imElement) => {

  //var t0 = performance.now();
  // var depth_time = 0;

  status_depth.textContent = 'Status: Loading image into model...';

  output_WIDTH = imElement.clientWidth;
  output_HEIGHT = imElement.clientHeight;

  // console.log(curr_model.concat("_").concat(curr_res))

  Depth_IMAGE_WIDTH = input_res_lookup[curr_model.concat("_").concat(curr_res)][1]
  Depth_IMAGE_HEIGHT = input_res_lookup[curr_model.concat("_").concat(curr_res)][0]

  // console.log(Depth_IMAGE_WIDTH)
  // console.log(Depth_IMAGE_HEIGHT)

  in_img = tf.browser.fromPixels(imElement).toFloat();

  const depthMask = tf.tidy(() => {

    const img = tf.image.resizeBilinear(in_img,
      [Depth_IMAGE_HEIGHT, Depth_IMAGE_WIDTH]);

    const scale = tf.scalar(255.);

    const mean = tf.tensor3d(norm_lookup[curr_model][0], [1, 1, 3]);
    const std = tf.tensor3d(norm_lookup[curr_model][1], [1, 1, 3]);

    const normalised = img.div(scale).sub(mean).div(std);

    status_depth.textContent = 'Status: Model loaded! running inference';
    const batched = normalised.transpose([2, 0, 1]).expandDims();

    //var it0 = performance.now();
    const features = model_depth_encoder.predict(batched);
    const predictions = model_depth_decoder.predict(features);
    //var it1 = performance.now();
    //depth_time = (it1 - it0);
    // const depthPred = predictions[3].squeeze(0).transpose([1, 2, 0]);
    const depthPred = predictions[out_idx[curr_model]].squeeze(0).transpose([1, 2, 0]);

    return depthPred.sub(depthPred.min()).divNoNan(depthPred.max().sub(depthPred.min()));

  });

  const depthCanvas = document.getElementById('depth');
  depthCanvas.width = output_WIDTH;
  depthCanvas.height = output_HEIGHT;

  //var t1 = performance.now();
  //console.log("Call to depth took " + (t1 - t0) + " milliseconds.");

  const depthMask_resized = tf.image.resizeBilinear(depthMask, [output_HEIGHT, output_WIDTH])

  img_array = tf.image.resizeBilinear(in_img, [output_HEIGHT, output_WIDTH]).arraySync()
  depth_array = depthMask_resized.arraySync()

  await tf.browser.toPixels(depthMask_resized, depthCanvas);

  status_depth.textContent = "Status: Done!";
  //status_depth.textContent = "Status: Done! inference took " + (depth_time.toFixed(1)) + " milliseconds.";
  //console.log("before: ", tf.memory());

  //tf.disposeVariables();
  //console.log("after: ", tf.memory());

  const xy = custom_mgrid(img_array.length, img_array[0].length)
  createPointCloud(xy[0].arraySync(), xy[1].arraySync(), depth_array, img_array);

  in_img.dispose();
  depthMask.dispose();
  depthMask_resized.dispose();
};


// model type settings

scene_indoors.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  curr_model = "indoors"
  update_model(curr_model);
  dropdown_scene.textContent = scene_indoors.textContent;
}
scene_outdoors1.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  curr_model = "outdoors1"
  update_model(curr_model);
  dropdown_scene.textContent = scene_outdoors1.textContent;
}
scene_outdoors2.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  curr_model = "outdoors2"
  update_model(curr_model);
  dropdown_scene.textContent = scene_outdoors2.textContent;
}

// input resolution settings

depth_low.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  curr_res = "low";

  dropdown_depth_qual.textContent = depth_low.textContent;
}
depth_medium.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  curr_res = "mid";
  dropdown_depth_qual.textContent = depth_medium.textContent;
}
depth_high.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  curr_res = "high";
  dropdown_depth_qual.textContent = depth_high.textContent;
}

const inpimg_lookup = {
  inp_outdoors: "assets/demo_images/inp_outdoors.png",
  inp_indoors: "assets/demo_images/inp_indoors.jfif"
}

function inp_load(inpimg) {
  event.preventDefault();
  // console.log(inpimg)
  document.getElementById('dropdown_input_out').textContent =
    document.getElementById(inpimg).textContent;
  document.getElementById("inpimg").src = inpimg_lookup[inpimg];
}

var filecheckBox = document.getElementById("fileinput");
var urlcheckBox = document.getElementById("urlinput");
var inp_selectcheckBox = document.getElementById("selectinput");
var filecontainer = document.getElementById("file-container");
var urlcontainer = document.getElementById("url-container");
var dropdown_input = document.getElementById("dropdown_input");

inp_selectcheckBox.addEventListener('click', function() {
  if (inp_selectcheckBox.checked == true) {
    filecheckBox.checked = false;
    urlcheckBox.checked = false;
    dropdown_input.style.display = "block";
    filecontainer.style.display = "none";
    urlcontainer.style.display = "none";
  }
});
filecheckBox.addEventListener('click', function() {
  if (filecheckBox.checked == true) {
    urlcheckBox.checked = false;
    inp_selectcheckBox.checked = false;
    filecontainer.style.display = "block";
    urlcontainer.style.display = "none";
    dropdown_input.style.display = "none";
  }
});

urlcheckBox.addEventListener('click', function() {
  if (urlcheckBox.checked == true) {
    filecheckBox.checked = false;
    inp_selectcheckBox.checked = false;
    filecontainer.style.display = "none";
    urlcontainer.style.display = "block";
    dropdown_input.style.display = "none";
  }
});

DepthWarmup();
