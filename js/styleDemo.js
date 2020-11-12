tf.setBackend('webgl');
// tf.ENV.set('WEBGL_CONV_IM2COL', false);
// tf.ENV.set('WEBGL_PACK', false); // This needs to be done otherwise things run very slow v1.0.4
// tf.webgl.forceHalfFloat();

//tf.enableDebugMode()
//tf.ENV.set('BEFORE_PAGING_CONSTANT ', 1000);
//tf.setBackend('cpu');
//tf.enableProdMode();

class MirrorPad extends tf.layers.Layer {
   static className = 'MirrorPad';

   constructor(config) {
     super(config);
   // config.mode = 'reflect'
   // console.log(config)
   // console.log("config.padding: ", config.padding)
   this.pad0 = config.padding[0]
   this.pad1 = config.padding[1]
   }
   call(inputs, kwargs) {
        // console.log(inputs)
        // console.log("this.pad0: ", this.pad0)
        // console.log("this.pad1: ", this.pad1)
        // let input = inputs;
        // if (Array.isArray(input)) {
        //     input = input[0];
        // }
        return inputs[0].mirrorPad([[0,0], this.pad0,
        this.pad1, [0,0]], 'reflect')
    }
}

tf.serialization.registerClass(MirrorPad);

let model_sem_encoder;
let model_sem_decoder;
let blur_kernel;
let model_transformer;

var segmentation_IMAGE_HEIGHT = 512
var segmentation_IMAGE_WIDTH = 512

// var style_IMAGE_HEIGHT = 384
// var style_IMAGE_WIDTH = 384
// const ratio = 0.75
const ratio = 0.5625
// const ratio = 1
// var style_IMAGE_WIDTH = 64
// var style_IMAGE_HEIGHT = Math.floor(style_IMAGE_WIDTH*(3/4))
var style_IMAGE_HEIGHT = 512
var style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT*ratio)

var output_HEIGHT = 512;
var output_WIDTH = Math.floor(output_HEIGHT*ratio);

var videoHeight = 512;
var videoWidth = Math.floor(videoHeight*0.75);
// var videoWidth = Math.floor(videoHeight*ratio);

var de_canvas = document.getElementById('mask');

const mean = tf.tensor3d([0.485, 0.456, 0.406], [1, 1, 3]);
const std = tf.tensor3d([0.229, 0.224, 0.225], [1, 1, 3]);
const scale = tf.scalar(255.);

var cors_api_url = 'https://cors-anywhere.herokuapp.com/';

const dropdown_seg_qual = document.getElementById('dropdown_seg_qual');
const seg_low = document.getElementById('seg_low')
const seg_medium = document.getElementById('seg_medium')
const seg_high = document.getElementById('seg_high')

const dropdown_style_qual = document.getElementById('dropdown_style_qual');
const style_low = document.getElementById('style_low')
const style_medium = document.getElementById('style_medium')
const style_high = document.getElementById('style_high')
const style_vhigh = document.getElementById('style_vhigh')

const mobile = isMobile();

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
  var style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT*ratio)
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

const Load_style_model = async (style_type) => {

  if (style_type == 'madhubani') {
    model_transformer = await tf.loadLayersModel('/assets/tfjs_layers_style_lite_madhubani/model.json');
    // model_transformer = await tf.loadLayersModel('/assets/tfjs_layers_style_lite/model.json');
      // model_transformer = await tf.loadLayersModel('/assets/TransformerNet_literrvocmadhubani4_pruned/model.json');
  } else {
      // model_transformer = await tf.loadLayersModel('/assets/tfjs_layers_style_lite/model.json');
        model_transformer = await tf.loadLayersModel('/assets/TransformerNet_literrvocmosaic_pruned/model.json');
          // model_transformer = await tf.loadLayersModel('/assets/TransformerNet_literrvocmosaic_pruned_test/model.json');
        // model_transformer = await tf.loadLayersModel('/assets/TransformerNet_literrvocmosaic3_pruned/model.json');
  }

}

const StyleWarmup = async () => {

  status_style.textContent = 'Status: Loading...';

  Load_style_model(style_type)

  model_sem_encoder = await tf.loadLayersModel('/assets/tfjs_layers_sem_encoder_bi_quant/model.json');
  model_sem_decoder = await tf.loadLayersModel('/assets/tfjs_layers_sem_decoder_pruned_quant/model.json');
  blur_kernel = await tf.loadLayersModel('/assets/gaus_21_1/model.json');

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

  var seg_time = 0;
  var style_time = 0;

  //var t0 = performance.now();
  //var bt0 = performance.now();

  // console.log(style_IMAGE_HEIGHT, style_IMAGE_WIDTH);

    var tbt0 = performance.now();

  const masked_style_comp = tf.tidy(() => {

    //console.log(tf.memory ());
    // output_WIDTH = imElement.clientWidth;
    // output_HEIGHT = imElement.clientHeight;
    var img = tf.browser.fromPixels(imElement).toFloat(); //tf.image.resizeBilinear(tf.browser.fromPixels(imElement).toFloat(),
    //[512,512]);

    console.log(img.shape)
    output_HEIGHT = img.shape[0];
    output_WIDTH = img.shape[1];
    console.log(output_HEIGHT,output_WIDTH)
    console.log(style_IMAGE_HEIGHT, style_IMAGE_WIDTH)

    const style_in = tf.image.resizeBilinear(img, [style_IMAGE_HEIGHT, style_IMAGE_WIDTH], true).expandDims();
    // const style_in = tf.image.resizeNearestNeighbor(img, [style_IMAGE_HEIGHT, style_IMAGE_WIDTH], true).expandDims();
    // const style_in = img.expandDims();
    // console.log(style_in.shape)
    //var img = tf.browser.fromPixels(imElement).toFloat();

    //console.log(tf.memory ());

    status_style.textContent = 'Status: Model loaded! running inference';

    //var bt1 = performance.now();
    //console.log("Call to pre took " + (bt1 - bt0) + " milliseconds.");

    //console.log(tf.memory ());

    // var st0 = performance.now();
     const style = model_transformer.predict(style_in);

    //console.log(tf.memory ());

    //var postt0 = performance.now();

   const style_out = style.squeeze(0).clipByValue(0, 255).div(scale);
   // const style_out = style_in.squeeze(0).clipByValue(0, 255).div(scale);

   // var st1 = performance.now();
   // style_time = (st1 - st0)
   // console.log("Call to style took " + style_time + " milliseconds.");

    //var postt1 = performance.now();
    //console.log("Call to post took " + (postt1 - postt0) + " milliseconds.");


    const resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
      output_WIDTH], true)

    if (output_type == 'style_only') {
      // return style_out
      return resized_style
    }
    else {
      const segmentation_img = tf.image.resizeBilinear(img, [segmentation_IMAGE_HEIGHT, segmentation_IMAGE_WIDTH], true)
      const segmentation_in = segmentation_img.div(scale).sub(mean).div(std).expandDims();
      const ones = tf.ones([output_HEIGHT,output_WIDTH])
      // const ones = tf.onesLike(img)
      // var sst0 = performance.now();
      const features = model_sem_encoder.predict(segmentation_in);
      const input_feature = tf.image.resizeBilinear(features[4], [64, 64], true)
      const predictions = model_sem_decoder.predict(input_feature);
      // var sst1 = performance.now();
      // seg_time = (sst1 - sst0)
      // console.log("Call to seg took " + seg_time + " milliseconds.");

      //const out = tf.image.resizeNearestNeighbor(predictions[0],[512,512]).squeeze(0);
      const Sem_mask = tf.image.resizeBilinear(predictions, [output_HEIGHT,
        output_WIDTH
      ], true).squeeze(0).argMax(2); //.expandDims(2);

      const mask = ones.where(Sem_mask.equal(15),0).expandDims(0).expandDims(3)
      // console.log(mask.shape)
      const blur_mask = blur_kernel.predict(mask).squeeze(0).squeeze(2);
      const blur_mask_neg = blur_mask.sub(1).abs()
      // console.log(blur_mask.shape)
      const mask_stacked = tf.stack([blur_mask, blur_mask, blur_mask], 2);
      const mask_stacked_neg = tf.stack([blur_mask_neg, blur_mask_neg, blur_mask_neg], 2);
      // console.log(mask_stacked.shape)

      if (output_type == 'masked_style') {
        return resized_style.mul(mask_stacked).add(img.div(scale).mul(mask_stacked_neg))
      }
      else {
        return img.div(scale).mul(mask_stacked).add(resized_style.mul(mask_stacked_neg))
      }
    }

  });

  //console.log(tf.memory ());

  // const maskCanvas = document.getElementById('mask');
  // const maskCanvas = document.getElementById('mask');

  await tf.browser.toPixels(masked_style_comp, de_canvas);

  var tbt1 = performance.now();
  console.log(style_type);
  console.log("Call to tb took " + (tbt1 - tbt0) + " milliseconds.");


  status_style.textContent = "Status: Done!";
  // status_style.textContent = "Status: Done! inference took " + ((seg_time + style_time).toFixed(1)) + " milliseconds.";

  //var t1 = performance.now();
  // console.log("Call to mask_Demo took " + (t1 - t0) + " milliseconds.");

  masked_style_comp.dispose();
  //console.log("after: ", tf.memory());
  //console.log(tf.memory ());
};

const style_Demo_RT = async (imElement) => {
  // output_WIDTH = imElement.clientWidth;
  // output_HEIGHT = imElement.clientHeight;
  // console.log(output_WIDTH,output_HEIGHT);

  const masked_style_comp = tf.tidy(() => {

    //console.log(tf.memory ());
    var img = tf.browser.fromPixels(imElement).toFloat(); //tf.image.resizeBilinear(tf.browser.fromPixels(imElement).toFloat(),
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

      return style_out

  });

  var tbt0 = performance.now();

  await tf.browser.toPixels(tf.image.resizeBilinear(masked_style_comp,
    [output_HEIGHT, output_WIDTH]), de_canvas);
  // await tf.browser.toPixels(tf.image.resizeNearestNeighbor(masked_style_comp,
  //   [output_HEIGHT, output_WIDTH]), de_canvas);
  // await tf.browser.toPixels(masked_style_comp, de_canvas);

  var tbt1 = performance.now();
  console.log("Call to tb took " + (tbt1 - tbt0) + " milliseconds.");

  masked_style_comp.dispose();
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

  console.log(de_canvas.width, de_canvas.height)
    console.log(de_canvas)
  const ctx = de_canvas.getContext('2d');
  console.log(ctx);
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

  console.log(video)

  console.log("##############################")
  console.log(video.srcObject.getVideoTracks()[0].getSettings())
  console.log("##############################")

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

    t_Width = document.getElementById("mainopt").clientWidth
    t_Height = 384;
    maini.style.display = "none";
    document.getElementById("main").style.display = "block";
    mainv.style.display = "block";

    // // apectWidth = (4 / 3) * t_Height
    apectWidth = 0.75 * t_Height
    output_WIDTH = apectWidth > t_Width ? t_Width : apectWidth;
    output_HEIGHT = t_Height;

    console.log(output_WIDTH, output_HEIGHT)
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
      video.srcObject.getTracks().forEach(function(track) {
        track.stop();
      });
    }
    document.getElementById("main").style.display = "block";
    maini.style.display = "block";
    mainv.style.display = "none";
  } else {
    maini.style.display = "none";
    document.getElementById("main").style.display = "none";
  }
}

const dropdown_output = document.getElementById('dropdown_output');
const style_only = document.getElementById('style_only')
const masked_style = document.getElementById('masked_style')
const masked_style_inverted = document.getElementById('masked_style_inverted')
//const all = document.getElementById('all')
var output_type = 'style_only'

style_only.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  output_type = 'style_only'
  dropdown_output.textContent = style_only.textContent;
}
masked_style.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  output_type = 'masked_style'
  dropdown_output.textContent = masked_style.textContent;
}
masked_style_inverted.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  output_type = 'masked_style_inverted'
  dropdown_output.textContent = masked_style_inverted.textContent;
}

const dropdown_style = document.getElementById('dropdown_style');
const mosaic = document.getElementById('mosaic')
const madhubani = document.getElementById('madhubani')
var style_type = 'mosaic'

mosaic.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  style_type = 'mosaic'
  Load_style_model(style_type)
  dropdown_style.textContent = mosaic.textContent;
}
madhubani.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  style_type = 'madhubani'
  Load_style_model(style_type)
  dropdown_style.textContent = madhubani.textContent;
}

seg_low.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  segmentation_IMAGE_HEIGHT = 128
  segmentation_IMAGE_WIDTH = 128
  dropdown_seg_qual.textContent = seg_low.textContent;
}
seg_medium.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  segmentation_IMAGE_HEIGHT = 256
  segmentation_IMAGE_WIDTH = 256
  dropdown_seg_qual.textContent = seg_medium.textContent;
}
seg_high.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  segmentation_IMAGE_HEIGHT = 512
  segmentation_IMAGE_WIDTH = 512
  dropdown_seg_qual.textContent = seg_high.textContent;
}

style_low.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  style_IMAGE_HEIGHT = 192
  style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT*ratio)
  dropdown_style_qual.textContent = style_low.textContent;
}
style_medium.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  style_IMAGE_HEIGHT = 256
  style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT*ratio)
  dropdown_style_qual.textContent = style_medium.textContent;
}
style_high.onclick = function() {
  event.preventDefault();
  // console.log("btn pressed!")
  style_IMAGE_HEIGHT = 384
  style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT*ratio)
  dropdown_style_qual.textContent = style_high.textContent;
}
style_vhigh.onclick = function() {
  event.preventDefault();
  // console.log("btn pressed!")
  style_IMAGE_HEIGHT = 512
  style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT*ratio)
  dropdown_style_qual.textContent = style_vhigh.textContent;
}

StyleWarmup();

// tf.setBackend('wasm').then(() => StyleWarmup());
