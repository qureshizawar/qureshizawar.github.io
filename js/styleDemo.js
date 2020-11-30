tf.setBackend('webgl');
// tf.ENV.set('WEBGL_CONV_IM2COL', false);
// tf.ENV.set('WEBGL_PACK', false); 
tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true)

//tf.enableDebugMode()
//tf.ENV.set('BEFORE_PAGING_CONSTANT ', 1000);
//tf.setBackend('cpu');
tf.enableProdMode();

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

// let model_sem_encoder;
// let model_sem_decoder;
// let blur_kernel;
let model_transformer;

// var style_IMAGE_HEIGHT = 384
// var style_IMAGE_WIDTH = 384
// const ratio = 0.75
const ratio = 0.5625
// const ratio = 1
// var style_IMAGE_WIDTH = 64
// var style_IMAGE_HEIGHT = Math.floor(style_IMAGE_WIDTH*(3/4))
var style_IMAGE_HEIGHT = 512
var style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT * ratio)

var output_HEIGHT = 512;
var output_WIDTH = Math.floor(output_HEIGHT * ratio);

var videoHeight = 512;
var videoWidth = Math.floor(videoHeight * 0.75);
// var videoWidth = Math.floor(videoHeight*ratio);

var de_canvas = document.getElementById('mask');

const mean = tf.tensor3d([0.485, 0.456, 0.406], [1, 1, 3]);
const std = tf.tensor3d([0.229, 0.224, 0.225], [1, 1, 3]);
const scale = tf.scalar(255.);

const dropdown_style_qual = document.getElementById('dropdown_style_qual');

const mobile = isMobile();


var style_type = 'mosaic_small'

var mode = 'user' //'user'

function is_touch_device() {
  return 'ontouchstart' in window // works on most browsers
    ||
    'onmsgesturechange' in window; // works on ie10
}

if (!(is_touch_device()) && !(window.matchMedia('(max-device-width: 960px)').matches)) {
  // style_IMAGE_HEIGHT = 128//512
  // style_IMAGE_WIDTH = 128//512
  var style_IMAGE_HEIGHT = 512
  var style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT * ratio)
  dropdown_style_qual.textContent = style_vhigh.textContent;
  //console.log("Desktop detected!")
}

document.getElementById("files_style").addEventListener("change", function(evt) {
  file_infer(evt.target.files[0], document.getElementById('inpimg_style'),
    status_style, style_Demo);
});
document.getElementById("style_files_btn").addEventListener("click", function(evt) {
  file = document.getElementById("files_style").files[0];
  if (file == null) {
    status_style.textContent = 'Status: File not found';
  } else {
    file_infer(file, document.getElementById('inpimg_style'),
      status_style, style_Demo);
  }
});

document.getElementById("btn_style").addEventListener("click", function(evt) {
  url_infer(document.getElementById('imagename_style'), document.getElementById('inpimg_style'),
    status_style, style_Demo);
});

const model_lookup = {
  // mosaic_small: '/assets/TransformerNet_literrvocmosaic6_pruned/model.json',
  mosaic_small: '/assets/mosaic6_style_literr_pruned_quant/model.json',
  mosaic: '/assets/mosaic_style_lite_quant/model.json',
  madhubani: '/assets/madhubani4_style_literr_pruned_quant/model.json',
  sketch: '/assets/sketch_style_literr_pruned_quant/model.json',
  feathers: '/assets/feathers_style_literr_pruned_quant/model.json',
};

const Load_style_model = async (style_type) => {
  // if (model_transformer) {
  //   console.log("disposing model")
  //   model_transformer.dispose();
  // }
  model_transformer = await tf.loadLayersModel(model_lookup[style_type]);
}

const StyleWarmup = async () => {

  status_style.textContent = 'Status: Loading...';

  await Load_style_model(style_type)

  // Make a prediction through the locally hosted inpimg_style.jpg.
  let img = document.getElementById('inpimg_style');
  set_static_output_size(img);
  if (img.complete && img.naturalHeight !== 0) {
    style_Demo(img);
    // style_Demo_RT(img);
    img.style.display = '';
  } else {
    img.onload = () => {
      style_Demo(img);
      // style_Demo_RT(img);
      inpElement.style.display = '';
    }
  }

  document.getElementById('file-container').style.display = '';
};


const style_Demo = async (imElement) => {

  status_style.textContent = 'Status: Loading image into model...';
  // console.log(style_IMAGE_HEIGHT, style_IMAGE_WIDTH);

  // var tbt0 = performance.now();

  const style_output = tf.tidy(() => {

    //console.log(tf.memory ());
    const img = tf.browser.fromPixels(imElement).toFloat();

    // console.log(img.shape)
    output_HEIGHT = img.shape[0];
    output_WIDTH = img.shape[1];
    // console.log(output_HEIGHT, output_WIDTH)
    // console.log(style_IMAGE_HEIGHT, style_IMAGE_WIDTH)

    const style_in = tf.image.resizeBilinear(img, [style_IMAGE_HEIGHT, style_IMAGE_WIDTH], true).expandDims();
    // const style_in = tf.image.resizeNearestNeighbor(img, [style_IMAGE_HEIGHT, style_IMAGE_WIDTH], true).expandDims();
    // const style_in = img.expandDims();
    // console.log(style_in.shape)
    //var img = tf.browser.fromPixels(imElement).toFloat();

    //console.log(tf.memory ());

    status_style.textContent = 'Status: Model loaded! running inference';

    //console.log(tf.memory ());

    const style = model_transformer.predict(style_in);

    //console.log(tf.memory ());

    const style_out = style.squeeze(0).clipByValue(0, 255).div(scale);


    const resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
      output_WIDTH
    ], true)

    return resized_style

  });

  //console.log(tf.memory ());


  await tf.browser.toPixels(style_output, de_canvas);

  // var tbt1 = performance.now();
  // console.log(style_type);
  // console.log("Call to tb took " + (tbt1 - tbt0) + " milliseconds.");


  status_style.textContent = "Status: Done!";

  //var t1 = performance.now();
  // console.log("Call to mask_Demo took " + (t1 - t0) + " milliseconds.");

  style_output.dispose();
  //console.log("after: ", tf.memory());
  //console.log(tf.memory ());
};

const style_Demo_RT = async (imElement) => {
  // output_WIDTH = imElement.clientWidth;
  // output_HEIGHT = imElement.clientHeight;
  // console.log(output_WIDTH,output_HEIGHT);

  const style_output = tf.tidy(() => {

    //console.log(tf.memory ());
    const img = tf.browser.fromPixels(imElement).toFloat(); //tf.image.resizeBilinear(tf.browser.fromPixels(imElement).toFloat(),
    //[512,512]);

    const style_in = tf.image.resizeBilinear(img, [style_IMAGE_HEIGHT, style_IMAGE_WIDTH], true).expandDims();
    // const style_in = tf.image.resizeNearestNeighbor(img, [style_IMAGE_HEIGHT, style_IMAGE_WIDTH], true).expandDims();
    // const style_in = img.expandDims();
    // console.log(img.shape)
    //var img = tf.browser.fromPixels(imElement).toFloat();

    const style = model_transformer.predict(style_in);

    const style_out = style.squeeze(0).clipByValue(0, 255).div(scale);
    // const style_out = style_in.squeeze(0).clipByValue(0, 255).div(scale);
    // const style_out = style_in.toInt();

    // style_out = tf.image.resizeBilinear(style_out,
    //   [output_HEIGHT, output_WIDTH])

    // return style_out

    const resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
      output_WIDTH
    ], true)

    return resized_style

  });

  // var tbt0 = performance.now();

  // await tf.browser.toPixels(tf.image.resizeBilinear(style_output,
  //   [output_HEIGHT, output_WIDTH]), de_canvas);
  // await tf.browser.toPixels(tf.image.resizeNearestNeighbor(masked_style_comp,
  //   [output_HEIGHT, output_WIDTH]), de_canvas);
  await tf.browser.toPixels(style_output, de_canvas);

  // var tbt1 = performance.now();
  // console.log("Call to tb took " + (tbt1 - tbt0) + " milliseconds.");

  style_output.dispose();
};

let request;

/**
 * Feeds an image to network to do inference - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectInRealTime(video) {
  // console.log("running detectInRealTime!")
  // const canvas = document.getElementById('output');
  de_canvas = document.getElementById('output');

  const flipHorizontal = mode == 'rear' ? false : true;

  de_canvas.width = output_WIDTH;
  de_canvas.height = output_HEIGHT;

  // console.log(de_canvas.width, de_canvas.height)
  // console.log(de_canvas)
  const ctx = de_canvas.getContext('2d');
  // console.log(ctx);
  ctx.save();

  async function DetectionFrame() {

    // console.log("running DetectionFrame!")

    // Begin monitoring code for frames per second
    //stats.begin();
    //t0 = performance.now();

    // ctx.clearRect(0, 0, output_WIDTH, output_HEIGHT);
    video.onloadeddata = () => {
      camloaded = true;
    }

    if (camloaded) {
      /*const time = await tf.time(() => style_Demo(video));
      console.log(`kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`);*/

      //await tf.nextFrame();
      // style_Demo(video);
      // console.log(video)
      style_Demo_RT(video);
      // await style_Demo_RT(video);

      // ctx.save();
      if (flipHorizontal) {
        ctx.scale(-1, 1);
        ctx.translate(-videoWidth, 0);
      }
      /*else{
        ctx.scale(1, 1);*/
      //console.log(video)
      // ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      // ctx.restore();

    }

    /*if (document.getElementById("show_fps").checked) {
      setupFPS();
      //document.getElementById('main').replaceChild(stats.dom, document.getElementById('fps'));
    }*/

    // End monitoring code for frames per second
    //stats.end();

    //console.log("DetectionFrame: ", performance.now() - t0)

    request = requestAnimationFrame(DetectionFrame);

  }

  DetectionFrame();
}

var filecheckBox = document.getElementById("fileinput");
var urlcheckBox = document.getElementById("urlinput");
var filecontainer = document.getElementById("file-container");
var urlcontainer = document.getElementById("url-container");

filecheckBox.addEventListener('click', function() {
  if (filecheckBox.checked == true) {
    urlcheckBox.checked = false;
    filecontainer.style.display = "block";
    urlcontainer.style.display = "none";
  } else {
    urlcheckBox.checked = true;
    filecontainer.style.display = "none";
    urlcontainer.style.display = "block";
  }
});

urlcheckBox.addEventListener('click', function() {
  if (urlcheckBox.checked == true) {
    filecheckBox.checked = false;
    filecontainer.style.display = "none";
    urlcontainer.style.display = "block";
  } else {
    filecheckBox.checked = true;
    filecontainer.style.display = "block";
    urlcontainer.style.display = "none";

  }
});

let video;

/**
 * Kicks off the demo by loading the model, finding and loading
 * available camera devices, and setting off the detectInRealTime function.
 */
async function bindPage() {
  //toggleLoadingUI(true);
  //toggleLoadingUI(false);

  camloaded = false;

  try {
    video = await loadVideo(mode);
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
      'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  // console.log(video)
  //
  // console.log("##############################")
  // console.log(video.srcObject.getVideoTracks()[0].getSettings())
  // console.log("##############################")

  //setupFPS();
  detectInRealTime(video);
}

function setmode() {
  if (document.getElementById("camMode").checked == true) {
    mode = 'rear';
  } else {
    mode = 'user';
  }

  if (document.getElementById("webcamc").checked == true) {
    video.srcObject.getTracks().forEach(function(track) {
      track.stop();
    });

    cancelAnimationFrame(request);
    //console.log("camMode");
    bindPage()
  }
}


function webc() {
  var imagecheckBox = document.getElementById("imagec");
  var videocheckBox = document.getElementById("webcamc");
  var maini = document.getElementById("mainImage");
  var mainv = document.getElementById("mainVideo");

  if (videocheckBox.checked == true) {
    document.getElementById("camswitch").style.display = "block";
    //if (mobile){document.getElementById("camswitch").style.display = "block";}
    imagecheckBox.checked = false;

    document.getElementById('dropdown_style_qual').textContent =
      document.getElementById("style_high").textContent;
    style_IMAGE_HEIGHT = style_res_lookup["style_high"]
    style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT * ratio)

    t_Width = document.getElementById("mainopt").clientWidth
    t_Height = 384;
    maini.style.display = "none";
    document.getElementById("main").style.display = "block";
    mainv.style.display = "block";

    // // apectWidth = (4 / 3) * t_Height
    apectWidth = 0.75 * t_Height
    output_WIDTH = apectWidth > t_Width ? t_Width : apectWidth;
    output_HEIGHT = t_Height;

    // console.log(output_WIDTH, output_HEIGHT)
    // videoHeight = 256;
    // videoWidth = Math.floor(videoHeight*ratio);

    // output_WIDTH = videoWidth
    // output_HEIGHT = videoHeight;

    camloaded = false;
    bindPage();
  } else {
    //console.log(video.srcObject.getTracks())

    document.getElementById("camswitch").style.display = "none";
    cancelAnimationFrame(request);
    video.srcObject.getTracks().forEach(function(track) {
      track.stop();
    });

    const ctx = document.getElementById('output').getContext('2d');
    ctx.clearRect(0, 0, output_WIDTH, output_HEIGHT);
    //console.log(video.srcObject.getTracks())
    mainv.style.display = "none";
    document.getElementById("main").style.display = "none";
  }
}

function imagec() {
  var imagecheckBox = document.getElementById("imagec");
  var videocheckBox = document.getElementById("webcamc");
  var maini = document.getElementById("mainImage");
  var mainv = document.getElementById("mainVideo");

  if (imagecheckBox.checked == true) {
    if (videocheckBox.checked == true) {
      document.getElementById("camswitch").style.display = "none";
      videocheckBox.checked = false;
      cancelAnimationFrame(request);
      if (video) {
        video.srcObject.getTracks().forEach(function(track) {
          track.stop();
        });
      }
    }
    document.getElementById("main").style.display = "block";
    maini.style.display = "block";
    mainv.style.display = "none";
  } else {
    maini.style.display = "none";
    document.getElementById("main").style.display = "none";
  }
}

function load_style(style) {
  event.preventDefault();
  document.getElementById('dropdown_style').textContent =
    document.getElementById(style).textContent;
  style_type = style
  Load_style_model(style_type)

}

const style_res_lookup = {
  style_low: 192,
  style_medium: 256,
  style_high: 384,
  style_vhigh: 512
}

function set_style_res(res) {
  event.preventDefault();
  document.getElementById('dropdown_style_qual').textContent =
    document.getElementById(res).textContent;
  style_IMAGE_HEIGHT = style_res_lookup[res]
  style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT * ratio)
}

StyleWarmup();

// tf.setBackend('wasm').then(() => StyleWarmup());
