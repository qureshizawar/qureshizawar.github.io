tf.setBackend('webgl');
tf.ENV.set('WEBGL_CONV_IM2COL', false);
tf.ENV.set('WEBGL_PACK', false); // This needs to be done otherwise things run very slow v1.0.4
tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true);
// tf.webgl.forceHalfFloat();

//tf.enableDebugMode()
//tf.ENV.set('BEFORE_PAGING_CONSTANT ', 1000);
//tf.setBackend('cpu');
//tf.enableProdMode();

download_img = function(el) {
  var image = document.getElementById("mask").toDataURL("image/jpg");
  el.href = image;
};

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
    return inputs[0].mirrorPad([
      [0, 0], this.pad0,
      this.pad1, [0, 0]
    ], 'reflect')
  }
}

tf.serialization.registerClass(MirrorPad);

let model_sem_encoder;
let model_sem_decoder;
let blur_kernel;

var blur_bg = false;
let model_transformer;

var segmentation_IMAGE_HEIGHT = 480
var segmentation_IMAGE_WIDTH = 480

// var segmentation_IMAGE_HEIGHT = 128
// var segmentation_IMAGE_WIDTH = 128

// var style_IMAGE_HEIGHT = 384
// var style_IMAGE_WIDTH = 384
var ratio = 0.75
// const ratio = 0.5625
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

var use_bg = true

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
  ratio = 1
  style_IMAGE_HEIGHT = 512
  style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT * ratio)
  dropdown_style_qual.textContent = style_vhigh.textContent;
  //console.log("Desktop detected!")
}

document.getElementById("bg_files").addEventListener("change", function(evt) {
  file_load(evt.target.files[0], document.getElementById('bg_img'),
    status_style);
});

document.getElementById("bg_url").addEventListener("click", function(evt) {
  url_load(document.getElementById('bg_imagename'), document.getElementById('bg_img'),
    status_style);
});

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

  if (model_transformer) {
    console.log("disposing model")
    model_transformer.dispose();

  }

  if (style_type == 'madhubani') {
    // model_transformer = await tf.loadLayersModel('/assets/tfjs_layers_style_lite_madhubani/model.json');
    model_transformer = await tf.loadLayersModel('/assets/TransformerNet_literrvocmadhubani4_pruned/model.json');
  } else if (style_type == 'mosaic_small') {
    // model_transformer = await tf.loadLayersModel('/assets/tfjs_layers_style_lite/model.json');
    // model_transformer = await tf.loadLayersModel('/assets/TransformerNet_literrvocmosaic_pruned/model.json');
    // model_transformer = await tf.loadLayersModel('/assets/TransformerNet_literrvocmosaic_pruned_test/model.json');
    model_transformer = await tf.loadLayersModel('/assets/TransformerNet_literrvocmosaic5_pruned/model.json');
  } else {
    model_transformer = await tf.loadLayersModel('/assets/tfjs_layers_style_lite/model.json');
  }

}

const StyleWarmup = async () => {

  status_style.textContent = 'Status: Loading...';

  Load_style_model(style_type)

  model_sem_encoder = await tf.loadLayersModel('/assets/tfjs_layers_sem_encoder_bi_quant/model.json');
  model_sem_decoder = await tf.loadLayersModel('/assets/tfjs_layers_sem_decoder_pruned_quant/model.json');
  // blur_kernel = await tf.loadLayersModel('/assets/gaus_11/model.json');
  blur_kernel = await tf.loadLayersModel('/assets/gaus_21_1/model.json');
  // blur_kernel = await tf.loadLayersModel('/assets/gaus_31/model.json');
  // blur_kernel = await tf.loadLayersModel('/assets/gaus_61/model.json');

  bg_blur_kernel = await tf.loadLayersModel('/assets/gaus_21_3/model.json');

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
  document.getElementById('download').style.display = "none";
  let bgElement = document.getElementById('bg_img');

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
    var bg_loaded = false;
    if (use_bg && bgElement.complete && bgElement.naturalHeight !== 0) {
      bg = tf.browser.fromPixels(bgElement).toFloat();
      const bg_resized_style = null
      bg_loaded = true;
      console.log(bg.shape)
    }


    console.log(img.shape)
    output_HEIGHT = img.shape[0];
    output_WIDTH = img.shape[1];
    console.log(output_HEIGHT, output_WIDTH)

    // if (output_type == 'masked_background' && use_bg !== true && bg_loaded !== true) {
    //   return img.div(scale)
    // }

    if (output_type == 'style_only' && use_bg == false && blur_bg == false) {

      console.log(style_IMAGE_HEIGHT, style_IMAGE_WIDTH)

      const style_in = tf.image.resizeBilinear(img, [style_IMAGE_HEIGHT, style_IMAGE_WIDTH], true).expandDims();

      status_style.textContent = 'Status: Model loaded! running inference';

      const style = model_transformer.predict(style_in);
      const style_out = style.squeeze(0).clipByValue(0, 255).div(scale);
      const resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
        output_WIDTH
      ], true)
      return resized_style

    } else {

      // console.log("segmentation needed")
      // console.log(output_type, use_bg, blur_bg)

      const segmentation_img = tf.image.resizeBilinear(img, [segmentation_IMAGE_HEIGHT, segmentation_IMAGE_WIDTH], true)
      const segmentation_in = segmentation_img.div(scale).sub(mean).div(std).expandDims();
      const ones = tf.ones([output_HEIGHT, output_WIDTH])

      const features = model_sem_encoder.predict(segmentation_in);
      const input_feature = tf.image.resizeNearestNeighbor(features[4], [64, 64])
      // const input_feature = tf.image.resizeBilinear(features[4], [64, 64], true)
      // const input_feature = tf.image.resizeBilinear(features[4], [64, 64], false)
      const predictions = model_sem_decoder.predict(input_feature);

      //const out = tf.image.resizeNearestNeighbor(predictions[0],[512,512]).squeeze(0);
      // const Sem_mask = tf.image.resizeNearestNeighbor(predictions, [output_HEIGHT,
      //   output_WIDTH
      // ]).squeeze(0).argMax(2); //.expandDims(2);
      const Sem_mask = tf.image.resizeBilinear(predictions, [output_HEIGHT,
        output_WIDTH
      ], false).squeeze(0).argMax(2);

      const mask = ones.where(Sem_mask.equal(15), 0).expandDims(0).expandDims(3)
      // console.log(mask.shape)
      const blur_mask = blur_kernel.predict(mask).squeeze(0).squeeze(2);
      const blur_mask_neg = blur_mask.sub(1).abs()
      // console.log(blur_mask.shape)
      const mask_stacked = tf.stack([blur_mask, blur_mask, blur_mask], 2);
      const mask_stacked_neg = tf.stack([blur_mask_neg, blur_mask_neg, blur_mask_neg], 2);
      // console.log(mask_stacked.shape)

      if (output_type == 'masked_background' && use_bg == false && blur_bg == true) {
        // just replace background
        console.log('just blur background')
        const blurred_img_bg = bg_blur_kernel.predict(img.expandDims(0)).squeeze(0)
        // return img.div(scale).mul(mask_stacked).add(bg_resized.div(scale).mul(mask_stacked_neg))
        return (img.div(scale).mul(mask_stacked).add(blurred_img_bg.div(scale).mul(mask_stacked_neg))).clipByValue(0, 1)
      } else if (output_type == 'masked_background' && use_bg && bg_loaded == true) {
        // just replace background
        console.log('just replace background')
        bg_resized = tf.image.resizeBilinear(bg, [output_HEIGHT,
          output_WIDTH
        ], true)
        if (blur_bg) {
          bg_resized = bg_blur_kernel.predict(bg_resized.expandDims(0)).squeeze(0)
        }
        // return img.div(scale).mul(mask_stacked).add(bg_resized.div(scale).mul(mask_stacked_neg))
        return (img.div(scale).mul(mask_stacked).add(bg_resized.div(scale).mul(mask_stacked_neg))).clipByValue(0, 1)
      } else if (output_type == 'style_only') {

        img_in = tf.image.resizeBilinear(img, [style_IMAGE_HEIGHT,
          style_IMAGE_WIDTH
        ], true).expandDims();
        mask_stacked_in = tf.image.resizeBilinear(mask_stacked, [style_IMAGE_HEIGHT,
          style_IMAGE_WIDTH
        ], true)
        mask_stacked_neg_in = tf.image.resizeBilinear(mask_stacked_neg, [style_IMAGE_HEIGHT,
          style_IMAGE_WIDTH
        ], true).expandDims();

        if (use_bg && bg_loaded == true) {
          // replace background and stylise
          console.log('replace background and stylise')
          bg_in = tf.image.resizeBilinear(bg, [style_IMAGE_HEIGHT,
            style_IMAGE_WIDTH
          ], true)


          if (blur_bg) {
            bg_in = bg_blur_kernel.predict(bg_in.expandDims(0)).squeeze(0)
          }

          style_in = img_in.mul(mask_stacked_in).add(bg_in.mul(mask_stacked_neg_in))

          style_out = model_transformer.predict(style_in).squeeze(0).clipByValue(0, 255).div(scale);
          resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
            output_WIDTH
          ], true)
          return resized_style
        } else if (blur_bg) {
          // blur background and stylise
          console.log('blur background and stylise')
          bg_in = bg_blur_kernel.predict(img_in).squeeze(0)

          style_in = img_in.mul(mask_stacked_in).add(bg_in.mul(mask_stacked_neg_in))
          style_out = model_transformer.predict(style_in).squeeze(0).clipByValue(0, 255).div(scale);
          resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
            output_WIDTH
          ], true)

          return resized_style
        }

      } else if (output_type == 'masked_style') {
        style_in = tf.image.resizeBilinear(img, [style_IMAGE_HEIGHT,
          style_IMAGE_WIDTH
        ], true).expandDims();
        style_out = model_transformer.predict(style_in).squeeze(0).clipByValue(0, 255).div(scale);
        resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
          output_WIDTH
        ], true)
        if (use_bg && bg_loaded == true) {
          // replace background and stylise person
          console.log('replace background and stylise person')
          bg_resized = tf.image.resizeBilinear(bg, [output_HEIGHT,
            output_WIDTH
          ], true)
          if (blur_bg) {
            bg_resized = bg_blur_kernel.predict(bg_resized.expandDims(0)).squeeze(0)
          }
          // return resized_style.mul(mask_stacked).add(bg_resized.mul(mask_stacked_neg).div(scale))
          return (resized_style.mul(mask_stacked).add(bg_resized.mul(mask_stacked_neg).div(scale))).clipByValue(0, 1)
        } else {
          // dont replace background and stylise person
          console.log('dont replace background and stylise person')

          if (blur_bg) {
            img = bg_blur_kernel.predict(img.expandDims(0)).squeeze(0)
          }
          // return resized_style.mul(mask_stacked).add(img.mul(mask_stacked_neg).div(scale))
          return (resized_style.mul(mask_stacked).add(img.mul(mask_stacked_neg).div(scale))).clipByValue(0, 1)
        }
      } else if (output_type == 'masked_style_inverted') {
        if (use_bg && bg_loaded == true) {
          // replace and stylise background
          console.log('replace and stylise background')

          style_in = tf.image.resizeBilinear(bg, [style_IMAGE_HEIGHT,
            style_IMAGE_WIDTH
          ], true).expandDims();
          style_out = model_transformer.predict(style_in).squeeze(0).clipByValue(0, 255).div(scale);
          resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
            output_WIDTH
          ], true)
          if (blur_bg) {
            resized_style = bg_blur_kernel.predict(resized_style.expandDims(0)).squeeze(0)
          }
          // return img.div(scale).mul(mask_stacked).add(resized_style.mul(mask_stacked_neg))
          return (img.div(scale).mul(mask_stacked).add(resized_style.mul(mask_stacked_neg))).clipByValue(0, 1)
        } else {
          // dont replace and stylise background
          console.log('dont replace and stylise background')

          style_in = tf.image.resizeBilinear(img, [style_IMAGE_HEIGHT,
            style_IMAGE_WIDTH
          ], true).expandDims();
          style_out = model_transformer.predict(style_in).squeeze(0).clipByValue(0, 255).div(scale);
          resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
            output_WIDTH
          ], true)
          if (blur_bg) {
            resized_style = bg_blur_kernel.predict(resized_style.expandDims(0)).squeeze(0)
          }
          // return img.div(scale).mul(mask_stacked).add(resized_style.mul(mask_stacked_neg))
          return (img.div(scale).mul(mask_stacked).add(resized_style.mul(mask_stacked_neg))).clipByValue(0, 1)
        }
      }

    }

    return img.div(scale)


    // } else {
    //
    //
    //   return resized_style.mul(mask_stacked).add(img.div(scale).mul(mask_stacked_neg))
    // } else {
    //   if (use_bg && bg_loaded == true) {
    //     const bg_style_in = tf.image.resizeBilinear(bg, [style_IMAGE_HEIGHT, style_IMAGE_WIDTH], true).expandDims();
    //     const bg_style = model_transformer.predict(bg_style_in);
    //     const bg_style_out = bg_style.squeeze(0).clipByValue(0, 255).div(scale);
    //     bg_resized_style = tf.image.resizeBilinear(bg_style_out, [output_HEIGHT,
    //       output_WIDTH
    //     ], true)
    //     return img.div(scale).mul(mask_stacked).add(bg_resized_style.mul(mask_stacked_neg))
    //   }
    //
    // }
    // }

  });

  //console.log(tf.memory ());

  // const maskCanvas = document.getElementById('mask');
  // const maskCanvas = document.getElementById('mask');

  await tf.browser.toPixels(masked_style_comp, de_canvas);

  var tbt1 = performance.now();
  console.log(style_type);
  console.log("Call to tb took " + (tbt1 - tbt0) + " milliseconds.");


  status_style.textContent = "Status: Done!";
  document.getElementById('download').style.display = "block";
  // status_style.textContent = "Status: Done! inference took " + ((seg_time + style_time).toFixed(1)) + " milliseconds.";

  //var t1 = performance.now();
  // console.log("Call to mask_Demo took " + (t1 - t0) + " milliseconds.");

  masked_style_comp.dispose();
  //console.log("after: ", tf.memory());
  //console.log(tf.memory ());
};

let request;

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

var bg_blurcheckBox = document.getElementById("blur_bg");
var bg_usecheckBox = document.getElementById("bg_use");
var bg_selectcheckBox = document.getElementById("bg_selectinput");
var bg_filecheckBox = document.getElementById("bg_fileinput");
var bg_urlcheckBox = document.getElementById("bg_urlinput");
var bg_dropdown = document.getElementById("dropdown_bg");
var bg_filecontainer = document.getElementById("bg_file-container");
var bg_urlcontainer = document.getElementById("bg_url-container");
var bg_urlbtn = document.getElementById("bg_url");


bg_blurcheckBox.addEventListener('click', function() {
  if (bg_blurcheckBox.checked == true) {
    blur_bg = true
  } else {
    blur_bg = false
  }
});
bg_usecheckBox.addEventListener('click', function() {
  if (bg_usecheckBox.checked == true) {
    document.getElementById("bg_table").style.display = "block";
    use_bg = true

  } else {
    document.getElementById("bg_table").style.display = "none";
    use_bg = false
  }
});

bg_selectcheckBox.addEventListener('click', function() {
  if (bg_selectcheckBox.checked == true) {
    bg_filecheckBox.checked = false;
    bg_urlcheckBox.checked = false;
    bg_dropdown.style.display = "block";
    bg_filecontainer.style.display = "none";
    bg_urlcontainer.style.display = "none";
    bg_urlbtn.style.display = "none";
  }
  // else {
  //   bg_urlcheckBox.checked = true;
  //   bg_filecontainer.style.display = "none";
  //   bg_urlcontainer.style.display = "block";
  //   bg_urlbtn.style.display = "block";
  // }
});

bg_filecheckBox.addEventListener('click', function() {
  if (bg_filecheckBox.checked == true) {
    bg_selectcheckBox.checked = false;
    bg_urlcheckBox.checked = false;
    bg_dropdown.style.display = "none";
    bg_filecontainer.style.display = "block";
    bg_urlcontainer.style.display = "none";
    bg_urlbtn.style.display = "none";
  }
  // else {
  //   bg_urlcheckBox.checked = true;
  //   bg_filecontainer.style.display = "none";
  //   bg_urlcontainer.style.display = "block";
  //   bg_urlbtn.style.display = "block";
  // }
});

bg_urlcheckBox.addEventListener('click', function() {
  if (bg_urlcheckBox.checked == true) {
    bg_selectcheckBox.checked = false;
    bg_filecheckBox.checked = false;
    bg_dropdown.style.display = "none";
    bg_filecontainer.style.display = "none";
    bg_urlcontainer.style.display = "block";
    bg_urlbtn.style.display = "block";
  }
  // else {
  //   bg_filecheckBox.checked = true;
  //   bg_filecontainer.style.display = "block";
  //   bg_urlcontainer.style.display = "none";
  //   bg_urlbtn.style.display = "none";
  // }
});


function bg_load(bg) {
  event.preventDefault();
  document.getElementById('bg_dropdown_out').textContent =
    document.getElementById(bg).textContent;
  document.getElementById("bg_img").src = "assets/demo_images/" + bg + ".jpg";
}

const dropdown_output = document.getElementById('dropdown_output');
const style_only = document.getElementById('style_only')
const masked_background = document.getElementById('masked_background')
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
masked_background.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  output_type = 'masked_background'
  dropdown_output.textContent = masked_background.textContent;
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

// const dropdown_style = document.getElementById('dropdown_style');
const mosaic = document.getElementById('mosaic')
const madhubani = document.getElementById('madhubani')
var style_type = 'mosaic'

function load_style(style) {
  event.preventDefault();
  document.getElementById('dropdown_style').textContent =
    document.getElementById(style).textContent;
  style_type = style
  Load_style_model(style_type)

}

// mosaic.onclick = function() {
//   event.preventDefault();
//   //console.log("btn pressed!")
//   style_type = 'mosaic'
//   Load_style_model(style_type)
//   dropdown_style.textContent = mosaic.textContent;
// }
// madhubani.onclick = function() {
//   event.preventDefault();
//   //console.log("btn pressed!")
//   style_type = 'madhubani'
//   Load_style_model(style_type)
//   dropdown_style.textContent = madhubani.textContent;
// }

seg_low.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  segmentation_IMAGE_HEIGHT = 256
  segmentation_IMAGE_WIDTH = 256
  dropdown_seg_qual.textContent = seg_low.textContent;
}
seg_medium.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  segmentation_IMAGE_HEIGHT = 380
  segmentation_IMAGE_WIDTH = 380
  dropdown_seg_qual.textContent = seg_medium.textContent;
}
seg_high.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  segmentation_IMAGE_HEIGHT = 480
  segmentation_IMAGE_WIDTH = 480
  dropdown_seg_qual.textContent = seg_high.textContent;
}

style_low.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  style_IMAGE_HEIGHT = 192
  style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT * ratio)
  dropdown_style_qual.textContent = style_low.textContent;
}
style_medium.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  style_IMAGE_HEIGHT = 256
  style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT * ratio)
  dropdown_style_qual.textContent = style_medium.textContent;
}
style_high.onclick = function() {
  event.preventDefault();
  // console.log("btn pressed!")
  style_IMAGE_HEIGHT = 384
  style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT * ratio)
  dropdown_style_qual.textContent = style_high.textContent;
}
style_vhigh.onclick = function() {
  event.preventDefault();
  // console.log("btn pressed!")
  style_IMAGE_HEIGHT = 512
  style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT * ratio)
  dropdown_style_qual.textContent = style_vhigh.textContent;
}

StyleWarmup();

// tf.setBackend('wasm').then(() => StyleWarmup());
