tf.setBackend('webgl');
tf.ENV.set('WEBGL_CONV_IM2COL', false);
tf.ENV.set('WEBGL_PACK', false);
tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true)

//tf.enableDebugMode()
// tf.ENV.set('BEFORE_PAGING_CONSTANT ', 1000);
//tf.setBackend('cpu');
tf.enableProdMode();

// console.log(tf.env())


let model_sem_encoder;
let model_sem_decoder;
let blur_kernel;

var blur_bg = false;
let model_transformer;

var segmentation_IMAGE_HEIGHT = 448
var segmentation_IMAGE_WIDTH = 448

// var ratio = 0.75
var ratio = 0.5625
// var ratio = 1
// var style_IMAGE_WIDTH = 64
// var style_IMAGE_HEIGHT = Math.floor(style_IMAGE_WIDTH*(3/4))
var style_IMAGE_HEIGHT = 512
var style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT * ratio)

var output_HEIGHT = 512;
var output_WIDTH = Math.floor(output_HEIGHT * ratio);

var use_bg = true

var de_canvas = document.getElementById('mask');

// const mean = tf.tensor3d([0.485, 0.456, 0.406], [1, 1, 3]);
// const std = tf.tensor3d([0.229, 0.224, 0.225], [1, 1, 3]);
// const scale = tf.scalar(255.);

const mobile = isMobile();

var mode = 'user' //'user'

if (!(is_touch_device()) && !(window.matchMedia('(max-device-width: 960px)').matches)) {
  // segmentation_IMAGE_HEIGHT = 448
  // segmentation_IMAGE_WIDTH = 448
  ratio = 1
  style_IMAGE_HEIGHT = 512
  style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT * ratio)
  dropdown_style_qual.textContent = style_high.textContent;
  // dropdown_style_qual.textContent = style_vhigh.textContent;
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

const model_lookup = {
  mosaic_small: '/assets/mosaic6_style_literr_pruned_quant/model.json',
  mosaic: '/assets/mosaic_style_lite_quant/model.json',
  madhubani: '/assets/madhubani4_style_literr_pruned_quant/model.json',
  sketch: '/assets/sketch_style_literr_pruned_quant/model.json',
  feathers: '/assets/feathers_style_literr_pruned_quant/model.json',
  oil: '/assets/oil1_style_literr_pruned_quant/model.json',
  // oil1: '/assets/oil1_style_literr_pruned_quant/model.json',
};
var curr_style
const Load_style_model = async (style_type) => {
  curr_style = style_type
  model_transformer = await tf.loadLayersModel(model_lookup[style_type]);
}

const StyleWarmup = async () => {

  status_style.textContent = 'Status: Loading...';

  Load_style_model(style_type)

  model_seg = await tf.loadLayersModel('/assets/mobile3_seg_bi_quant/model.json');
  // blur_kernel = await tf.loadLayersModel('/assets/gaus_11/model.json');
  blur_kernel = await tf.loadLayersModel('/assets/gaus_21_1/model.json');
  bg_blur_kernel = await tf.loadLayersModel('/assets/gaus_21_3/model.json');

  // Make a prediction through the locally hosted inpimg_style.jpg.
  let img = document.getElementById('inpimg_style');
  set_static_output_size(img);
  if (img.complete && img.naturalHeight !== 0) {
    style_Demo(img);
    img.style.display = '';
  } else {
    img.onload = () => {
      style_Demo(img);
      inpElement.style.display = '';
    }
  }

  document.getElementById('file-container').style.display = '';
};


const style_Demo = async (imElement) => {

  status_style.textContent = 'Status: Loading image into model...';
  document.getElementById('download').style.display = "none";
  let bgElement = document.getElementById('bg_img');

  // var seg_time = 0;
  // var style_time = 0;

  //var t0 = performance.now();
  // var bt0 = performance.now();

  // console.log(style_IMAGE_HEIGHT, style_IMAGE_WIDTH);

  // var tbt0 = performance.now();

  var masked_style_comp = tf.tidy(() => {

    const mean = tf.tensor3d([0.485, 0.456, 0.406], [1, 1, 3]);
    const std = tf.tensor3d([0.229, 0.224, 0.225], [1, 1, 3]);
    const scale = tf.scalar(255.);

    // console.log(tf.memory());
    // output_WIDTH = imElement.clientWidth;
    // output_HEIGHT = imElement.clientHeight;
    var img = tf.browser.fromPixels(imElement).toFloat(); //tf.image.resizeBilinear(tf.browser.fromPixels(imElement).toFloat(),
    //[512,512]);
    var bg_loaded = false;
    if (use_bg && bgElement.complete && bgElement.naturalHeight !== 0) {
      bg = tf.browser.fromPixels(bgElement).toFloat();
      const bg_resized_style = null
      bg_loaded = true;
      // console.log(bg.shape)
    }


    // console.log(img.shape)
    output_HEIGHT = img.shape[0];
    output_WIDTH = img.shape[1];
    // console.log(output_HEIGHT, output_WIDTH)

    // if (output_type == 'masked_background' && use_bg !== true && bg_loaded !== true) {
    //   return img.div(scale)
    // }

    if (output_type == 'style_only' && use_bg == false && blur_bg == false) {

      // console.log(style_IMAGE_HEIGHT, style_IMAGE_WIDTH)

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
      // const segmentation_in = segmentation_img.div(scale).sub(mean).div(std).expandDims();
      const segmentation_in = segmentation_img.div(scale).sub(mean).div(std).transpose([2, 0, 1]).expandDims();
      const ones = tf.ones([output_HEIGHT, output_WIDTH])

      const predictions = model_seg.predict(segmentation_in).transpose([0, 2, 3, 1]);

      const Sem_mask = tf.image.resizeBilinear(predictions, [output_HEIGHT,
        output_WIDTH
      ], false).squeeze(0).argMax(2);
      // console.log(Sem_mask)
      // const mask = ones.where(Sem_mask.equal(15), 0).expandDims(0).expandDims(3)
      const mask = ones.where(Sem_mask.equal(1), 0).expandDims(0).expandDims(3)
      // console.log(mask.shape)
      const blur_mask = blur_kernel.predict(mask).squeeze(0).squeeze(2);
      const blur_mask_neg = blur_mask.sub(1).abs()
      // console.log(blur_mask.shape)
      const mask_stacked = tf.stack([blur_mask, blur_mask, blur_mask], 2);
      const mask_stacked_neg = tf.stack([blur_mask_neg, blur_mask_neg, blur_mask_neg], 2);
      // console.log(mask_stacked.shape)

      if (output_type == 'masked_background' && use_bg == false && blur_bg == true) {
        // just replace background
        // console.log('just blur background')
        const blurred_img_bg = bg_blur_kernel.predict(img.expandDims(0)).squeeze(0)
        // return img.div(scale).mul(mask_stacked).add(bg_resized.div(scale).mul(mask_stacked_neg))
        return (img.div(scale).mul(mask_stacked).add(blurred_img_bg.div(scale).mul(mask_stacked_neg))).clipByValue(0, 1)
      } else if (output_type == 'masked_background' && use_bg && bg_loaded == true) {
        // just replace background
        // console.log('just replace background')
        var bg_resized = tf.image.resizeBilinear(bg, [output_HEIGHT,
          output_WIDTH
        ], true)
        if (blur_bg) {
          bg_resized = bg_blur_kernel.predict(bg_resized.expandDims(0)).squeeze(0)
        }
        // return img.div(scale).mul(mask_stacked).add(bg_resized.div(scale).mul(mask_stacked_neg))
        return (img.div(scale).mul(mask_stacked).add(bg_resized.div(scale).mul(mask_stacked_neg))).clipByValue(0, 1)
      } else if (output_type == 'style_only') {

        const img_in = tf.image.resizeBilinear(img, [style_IMAGE_HEIGHT,
          style_IMAGE_WIDTH
        ], true).expandDims();
        const mask_stacked_in = tf.image.resizeBilinear(mask_stacked, [style_IMAGE_HEIGHT,
          style_IMAGE_WIDTH
        ], true)
        const mask_stacked_neg_in = tf.image.resizeBilinear(mask_stacked_neg, [style_IMAGE_HEIGHT,
          style_IMAGE_WIDTH
        ], true).expandDims();

        if (use_bg && bg_loaded == true) {
          // replace background and stylise
          // console.log('replace background and stylise')
          var bg_in = tf.image.resizeBilinear(bg, [style_IMAGE_HEIGHT,
            style_IMAGE_WIDTH
          ], true)


          if (blur_bg) {
            bg_in = bg_blur_kernel.predict(bg_in.expandDims(0)).squeeze(0)
          }

          const style_in = img_in.mul(mask_stacked_in).add(bg_in.mul(mask_stacked_neg_in))

          const style_out = model_transformer.predict(style_in).squeeze(0).clipByValue(0, 255).div(scale);
          const resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
            output_WIDTH
          ], true)
          // console.log(tf.memory());
          return resized_style
        } else if (blur_bg) {
          // blur background and stylise
          // console.log('blur background and stylise')
          const bg_in = bg_blur_kernel.predict(img_in).squeeze(0)

          const style_in = img_in.mul(mask_stacked_in).add(bg_in.mul(mask_stacked_neg_in))
          const style_out = model_transformer.predict(style_in).squeeze(0).clipByValue(0, 255).div(scale);
          const resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
            output_WIDTH
          ], true)

          return resized_style
        }

      } else if (output_type == 'masked_style') {
        const style_in = tf.image.resizeBilinear(img, [style_IMAGE_HEIGHT,
          style_IMAGE_WIDTH
        ], true).expandDims();
        const style_out = model_transformer.predict(style_in).squeeze(0).clipByValue(0, 255).div(scale);
        const resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
          output_WIDTH
        ], true)
        if (use_bg && bg_loaded == true) {
          // replace background and stylise person
          // console.log('replace background and stylise person')
          var bg_resized = tf.image.resizeBilinear(bg, [output_HEIGHT,
            output_WIDTH
          ], true)
          if (blur_bg) {
            bg_resized = bg_blur_kernel.predict(bg_resized.expandDims(0)).squeeze(0)
          }
          // return resized_style.mul(mask_stacked).add(bg_resized.mul(mask_stacked_neg).div(scale))
          return (resized_style.mul(mask_stacked).add(bg_resized.mul(mask_stacked_neg).div(scale))).clipByValue(0, 1)
        } else {
          // dont replace background and stylise person
          // console.log('dont replace background and stylise person')

          if (blur_bg) {
            const img = bg_blur_kernel.predict(img.expandDims(0)).squeeze(0)
          }
          // return resized_style.mul(mask_stacked).add(img.mul(mask_stacked_neg).div(scale))
          return (resized_style.mul(mask_stacked).add(img.mul(mask_stacked_neg).div(scale))).clipByValue(0, 1)
        }
      } else if (output_type == 'masked_style_inverted') {
        if (use_bg && bg_loaded == true) {
          // replace and stylise background
          // console.log('replace and stylise background')

          const style_in = tf.image.resizeBilinear(bg, [style_IMAGE_HEIGHT,
            style_IMAGE_WIDTH
          ], true).expandDims();
          const style_out = model_transformer.predict(style_in).squeeze(0).clipByValue(0, 255).div(scale);
          var resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
            output_WIDTH
          ], true)
          if (blur_bg) {
            resized_style = bg_blur_kernel.predict(resized_style.expandDims(0)).squeeze(0)
          }
          // return img.div(scale).mul(mask_stacked).add(resized_style.mul(mask_stacked_neg))
          return (img.div(scale).mul(mask_stacked).add(resized_style.mul(mask_stacked_neg))).clipByValue(0, 1)
        } else {
          // dont replace and stylise background
          // console.log('dont replace and stylise background')

          const style_in = tf.image.resizeBilinear(img, [style_IMAGE_HEIGHT,
            style_IMAGE_WIDTH
          ], true).expandDims();
          const style_out = model_transformer.predict(style_in).squeeze(0).clipByValue(0, 255).div(scale);
          var resized_style = tf.image.resizeBilinear(style_out, [output_HEIGHT,
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
  });

  // console.log(tf.memory());

  if (output_type !== 'masked_background' && curr_style == "sketch") {
    masked_style_comp = tf.tidy(() => {
      const gray = tf.tensor3d([0.2126, 0.7152, 0.0722], [1, 1, 3]);
      return masked_style_comp.mul(gray).sum(2)
    });
  }


  await tf.browser.toPixels(masked_style_comp, de_canvas);

  var tbt1 = performance.now();
  // console.log(style_type);
  // console.log("Call to tb took " + (tbt1 - tbt0) + " milliseconds.");

  status_style.textContent = "Status: Done!";
  document.getElementById('download').style.display = "block";
  // status_style.textContent = "Status: Done! inference took " + ((seg_time + style_time).toFixed(1)) + " milliseconds.";

  //var t1 = performance.now();
  // console.log("Call to mask_Demo took " + (t1 - t0) + " milliseconds.");

  masked_style_comp.dispose();
  // tf.disposeVariables()
  //console.log("after: ", tf.memory());
  // console.log(tf.memory());
};


function bg_load(bg) {
  event.preventDefault();
  document.getElementById('bg_dropdown_out').textContent =
    document.getElementById(bg).textContent;
  document.getElementById("bg_img").src = "assets/demo_images/" + bg + ".jpg";
}

var output_type = 'style_only'

function set_output(outputType) {
  event.preventDefault();
  document.getElementById('dropdown_output').textContent =
    document.getElementById(outputType).textContent;
  output_type = outputType
}

var style_type = 'mosaic'

function load_style(style) {
  event.preventDefault();
  document.getElementById('dropdown_style').textContent =
    document.getElementById(style).textContent;
  style_type = style
  Load_style_model(style_type)
}

const seg_res_lookup = {
  seg_low: 256,
  seg_medium: 380,
  seg_high: 480
}

function set_seg_res(res) {
  event.preventDefault();
  document.getElementById('dropdown_seg_qual').textContent =
    document.getElementById(res).textContent;
  segmentation_IMAGE_HEIGHT = seg_res_lookup[res]
  segmentation_IMAGE_WIDTH = seg_res_lookup[res]
}

// const style_res_lookup = {style_low:192, style_medium:256, style_high:384,
//   style_vhigh:512}
const style_res_lookup = {
  style_low: 256,
  style_medium: 384,
  style_high: 512
}

function set_style_res(res) {
  event.preventDefault();
  document.getElementById('dropdown_style_qual').textContent =
    document.getElementById(res).textContent;
  style_IMAGE_HEIGHT = style_res_lookup[res]
  style_IMAGE_WIDTH = Math.floor(style_IMAGE_HEIGHT * ratio)
}

// let request;

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


download_img = function(el) {
  var image = document.getElementById("mask").toDataURL("image/jpg");
  el.href = image;
};

function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

class MirrorPad extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.pad0 = config.padding[0];
    this.pad1 = config.padding[1];
  }

  call(inputs, kwargs) {
    return inputs[0].mirrorPad([[0, 0], this.pad0, this.pad1, [0, 0]], 'reflect');
  }

}

_defineProperty(MirrorPad, "className", 'MirrorPad');

tf.serialization.registerClass(MirrorPad);

// StyleWarmup();

// tf.setBackend('wasm').then(() => StyleWarmup());
