tf.setBackend('webgl');
tf.ENV.set('WEBGL_CONV_IM2COL', false);
tf.ENV.set('WEBGL_PACK', false);
tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true)

tf.enableProdMode();

let model_depth;

// H x W
const input_res_lookup = {
  scene_outdoors1_depth_low: [128, 320],
  scene_outdoors1_depth_mid: [192, 320],
  scene_outdoors1_depth_high: [192, 640],
  scene_outdoors2_depth_low: [160, 320],
  scene_outdoors2_depth_mid: [256, 480],
  scene_outdoors2_depth_high: [320, 640],
  scene_indoors_depth_low: [224, 320],
  scene_indoors_depth_mid: [320, 448],
  scene_indoors_depth_high: [416, 544]
};

var curr_model = "scene_outdoors2"

// var curr_res = "depth_high"
// var Depth_IMAGE_HEIGHT = 320;
// var Depth_IMAGE_WIDTH = 640;

var curr_res = "depth_mid"
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
  curr_res = "depth_high"
  dropdown_depth_qual.textContent = depth_high.textContent;
  Depth_IMAGE_WIDTH = input_res_lookup[curr_model.concat("_").concat(curr_res)][1]
  Depth_IMAGE_HEIGHT = input_res_lookup[curr_model.concat("_").concat(curr_res)][0]
}

document.getElementById("files").addEventListener("change", function(evt) {
  status_depth.textContent = 'Status: Loading...';
  file_infer(evt.target.files[0], document.getElementById('inpimg'),
    status_depth, Depth_Demo);

  // status_depth.textContent = "Status: Done!";
});
document.getElementById("depth_files_btn").addEventListener("click", function(evt) {
  file = document.getElementById("files").files[0];
  if (file == null) {
    status_depth.textContent = 'Status: File not found';
  } else {
    status_depth.textContent = 'Status: Loading...';
    file_infer(file, document.getElementById('inpimg'),
      status_depth, Depth_Demo);
    // status_depth.textContent = "Status: Done!";
  }
});
document.getElementById("select_files_btn").onclick = function() {
  // event.preventDefault();
  // console.log("btn pressed!")
  status_depth.textContent = 'Status: Loading...';
  img_in = document.getElementById('inpimg')
  Depth_Demo(img_in);
  // status_depth.textContent = "Status: Done!";
}
// document.getElementById("depth_files_btn").addEventListener("click", function(evt) {
//   file = document.getElementById("files").files[0];
//   if (file == null) {
//     status_depth.textContent = 'Status: File not found';
//   } else {
//     file_infer(file, document.getElementById('inpimg'),
//       status_depth, Depth_Demo);
//   }
// });

document.getElementById("url_btn").addEventListener("click", function(evt) {
  status_depth.textContent = 'Status: Loading...';
  url_infer(document.getElementById('imagename'), document.getElementById('inpimg'),
    status_depth, Depth_Demo);
  // status_depth.textContent = "Status: Done!";
});

function custom_mgrid(rows, cols) {
  a = tf.tile(tf.range(0, rows).reshape([rows, 1]), [1, cols]); // cols
  b = tf.tile(tf.range(0, cols).expandDims(0), [rows, 1]); // rows
  return [a, b];
}

const load_model = async (name) => {
  // console.log("load_model!")
  if (name == "scene_indoors") {
    model_depth = await tf.loadLayersModel('/assets/indoors_Mnet_tfjs_quant/model.json');
  } else {
    model_depth = await tf.loadLayersModel('/assets/outdoors_Mnet_tfjs_quant/model.json');
  }

};

const DepthWarmup = async () => {

  status_depth.textContent = 'Status: Loading...';
  model_depth = await tf.loadLayersModel('/assets/outdoors_Mnet_tfjs_quant/model.json');
  // await load_model(curr_model);

  // Make a prediction through the locally hosted inpimg.jpg.
  inpElement = document.getElementById('inpimg');
  //inpElement.src = e.target.result;
  if (inpElement.complete && inpElement.naturalHeight !== 0) {
    // set_static_output_size(inpElement);
    Depth_Demo(inpElement);
    inpElement.style.display = '';
  } else {
    inpElement.onload = () => {
      // set_static_output_size(inpElement);
      Depth_Demo(inpElement);
      inpElement.style.display = '';
    }
  }

  // status_depth.textContent = "Status: Done!";
  // document.getElementById('file-container').style.display = '';
};

const Depth_Demo = async (imElement) => {

  // var t0 = performance.now();
  // var depth_time = 0;

  // status_depth.textContent = 'Status: Loading image into model...';
  status_depth.textContent = 'Status: Loading...';

  output_WIDTH = imElement.clientWidth;
  output_HEIGHT = imElement.clientHeight;

  // console.log(curr_model.concat("_").concat(curr_res))

  // console.log(Depth_IMAGE_WIDTH)
  // console.log(Depth_IMAGE_HEIGHT)

  // console.log("before: ", tf.memory());

  // in_img = tf.browser.fromPixels(imElement).toFloat();

  const depthMask_resized = tf.tidy(() => {


    in_img = tf.browser.fromPixels(imElement).toFloat();

    const img = tf.image.resizeBilinear(in_img,
      [Depth_IMAGE_HEIGHT, Depth_IMAGE_WIDTH]);

    const scale = tf.scalar(255.);

    const mean = tf.tensor3d([0.485, 0.456, 0.406], [1, 1, 3]);
    const std = tf.tensor3d([0.229, 0.224, 0.225], [1, 1, 3]);

    const normalised = img.div(scale).sub(mean).div(std);

    status_depth.textContent = 'Status: Model loaded! running inference';
    const batched = normalised.expandDims();

    const dispPred = model_depth.predict(batched).squeeze(0) //.transpose([1, 2, 0]);

    const dispPred_norm = dispPred.sub(dispPred.min()).divNoNan(dispPred.max().sub(dispPred.min()));

    img_array = tf.image.resizeBilinear(in_img, [output_HEIGHT, output_WIDTH]).arraySync()

    const xy = custom_mgrid(img_array.length, img_array[0].length)
    xs_array = xy[0].arraySync()
    ys_array = xy[1].arraySync()

    return tf.image.resizeBilinear(dispPred_norm, [output_HEIGHT, output_WIDTH])
  });

  const depthCanvas = document.getElementById('depth');
  depthCanvas.width = output_WIDTH;
  depthCanvas.height = output_HEIGHT;

  depth_array = depthMask_resized.arraySync()

  await tf.browser.toPixels(depthMask_resized, depthCanvas);

  //status_depth.textContent = "Status: Done! inference took " + (depth_time.toFixed(1)) + " milliseconds.";

  createPointCloud(xs_array, ys_array, depth_array, img_array);

  // var t1 = performance.now();
  // console.log("Call to depth took " + (t1 - t0) + " milliseconds.");

  // console.log("after: ", tf.memory());

  depthMask_resized.dispose();
  status_depth.textContent = "Status: Done!";
  // console.log("after: ", tf.memory());
};


// model type settings

function set_model(model) {
  event.preventDefault();
  curr_model = model
  load_model(curr_model);
  Depth_IMAGE_WIDTH = input_res_lookup[curr_model.concat("_").concat(curr_res)][1]
  Depth_IMAGE_HEIGHT = input_res_lookup[curr_model.concat("_").concat(curr_res)][0]

  dropdown_scene.textContent = document.getElementById(model).textContent;
}

// input resolution settings

function set_res(res) {
  event.preventDefault();
  curr_res = res;
  Depth_IMAGE_WIDTH = input_res_lookup[curr_model.concat("_").concat(curr_res)][1]
  Depth_IMAGE_HEIGHT = input_res_lookup[curr_model.concat("_").concat(curr_res)][0]
  dropdown_depth_qual.textContent = document.getElementById(res).textContent;
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

class MirrorPad extends tf.layers.Layer {
  static className = 'MirrorPad';

  constructor(config) {
    super(config);
    this.pad0 = config.padding[0]
    this.pad1 = config.padding[1]
  }
  call(inputs, kwargs) {
    return inputs[0].mirrorPad([
      [0, 0], this.pad0,
      this.pad1, [0, 0]
    ], 'reflect')
  }
}

tf.serialization.registerClass(MirrorPad);

// DepthWarmup();
