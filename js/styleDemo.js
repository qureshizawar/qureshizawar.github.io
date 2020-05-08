tf.setBackend('webgl')
//let backend = new tf.webgl.MathBackendWebGL()
tf.ENV.set('WEBGL_CONV_IM2COL', false);

//console.log(tf.ENV.features)
//tf.ENV.set('BEFORE_PAGING_CONSTANT ', 1000);
//tf.setBackend('cpu');
//tf.enableProdMode();

let model_sem_encoder;
let model_sem_decoder;
let model_transformer;

var segmentation_IMAGE_HEIGHT = 512
var segmentation_IMAGE_WIDTH = 512

var style_IMAGE_HEIGHT = 384
var style_IMAGE_WIDTH = 384

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

function is_touch_device() {
  return 'ontouchstart' in window // works on most browsers
    ||
    'onmsgesturechange' in window; // works on ie10
}
//console.log(screen.width)
//console.log((window.matchMedia('(max-device-width: 960px)').matches))
if (!(is_touch_device()) && !(window.matchMedia('(max-device-width: 960px)').matches)) {
  style_IMAGE_HEIGHT = 512
  style_IMAGE_WIDTH = 512
  dropdown_style_qual.textContent = style_vhigh.textContent;
  //console.log("Desktop detected!")
}

// see https://stackoverflow.com/questions/20600800/js-client-side-exif-orientation-rotate-and-mirror-jpeg-images
function style_file(image) {
  status_style.textContent = 'Status: Fetching image...';
  //var tt = performance.now();
  //var image = evt.target.files[0]; // FileList object
  if (window.File && window.FileReader && window.FileList && window.Blob) {
    var reader = new FileReader();
    // Closure to capture the file information.
    reader.addEventListener("load", function(e) {
      const imageData = e.target.result;
      //const imageElement = document.createElement("img");
      //imageElement.setAttribute("src", imageData);
      //document.getElementById("container1").appendChild(imageElement);
      window.loadImage(imageData, function(img) {
        if (img.type === "error") {
          console.log("couldn't load image:", img);
        } else {
          window.EXIF.getData(img, function() {
            //console.log("done!");
            var orientation = window.EXIF.getTag(this, "Orientation");
            var canvas = window.loadImage.scale(img, {
              orientation: orientation || 0,
              canvas: true
            });
            //document.getElementById("container2").appendChild(canvas);
            // or using jquery $("#container").append(canvas);
            let img_out = document.getElementById('inpimg_style');
            img_out.src = canvas.toDataURL();
            //console.log('orientation took: ');
            //console.log(performance.now()-tt);
            img_out.onload = () => style_Demo(img_out);
          });
        }
      });
    });
    reader.readAsDataURL(image);
  } else {
    console.log('The File APIs are not fully supported in this browser.');
  }
};

document.getElementById("files_style").addEventListener("change", function(evt) {
  style_file(evt.target.files[0]);
});
document.getElementById("style_files_btn").addEventListener("click", function(evt) {
  file = document.getElementById("files_style").files[0];
  if (file == null) {
    status_style.textContent = 'Status: File not found';
  } else {
    style_file(file);
  }
});

document.getElementById('btn_style').onclick = function() {
  status_style.textContent = 'Status: Fetching image...';
  let url = new URL(document.getElementById('imagename_style').value);
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
      let img = document.getElementById('inpimg_style');
      img.src = e.target.result;
      //img.height = IMAGE_HEIGHT;
      //img.width = IMAGE_WIDTH;
      img.onload = () => style_Demo(img);
    };
  };
}

const Load_style_model = async (style_type) => {

  if (style_type == 'madhubani') {
    model_transformer = await tf.loadLayersModel('/assets/tfjs_layers_style_lite_madhubani/model.json');
  } else {
    model_transformer = await tf.loadLayersModel('/assets/tfjs_layers_style_lite/model.json');
  }

}

const StyleWarmup = async () => {

  status_style.textContent = 'Status: Loading...';

  //console.log(tf.memory ());

  Load_style_model(style_type)

  model_sem_encoder = await tf.loadLayersModel('/assets/tfjs_layers_sem_encoder_bi_quant/model.json');
  model_sem_decoder = await tf.loadLayersModel('/assets/tfjs_layers_sem_decoder_pruned_quant/model.json');

  //model_sem_decoder.summary();

  //console.log(tf.memory());

  //console.log(tf.memory());

  // Make a prediction through the locally hosted inpimg_style.jpg.
  let inpElement = document.getElementById('inpimg_style');
  //inpElement.width = 300
  //inpElement.height = 300
  //inpElement.src = e.target.result;
  if (inpElement.complete && inpElement.naturalHeight !== 0) {
    style_Demo(inpElement);
    inpElement.style.display = '';
  } else {
    inpElement.onload = () => {
      style_Demo(inpElement);
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

  const masked_style_comp = tf.tidy(() => {

    //console.log(tf.memory ());

    var img = tf.browser.fromPixels(imElement).toFloat(); //tf.image.resizeBilinear(tf.browser.fromPixels(imElement).toFloat(),
    //[512,512]);

    const style_in = tf.image.resizeBilinear(img, [style_IMAGE_HEIGHT, style_IMAGE_WIDTH], true).expandDims();
    //var img = tf.browser.fromPixels(imElement).toFloat();

    //console.log(tf.memory ());

    const mean = tf.tensor3d([0.485, 0.456, 0.406], [1, 1, 3]);
    const std = tf.tensor3d([0.229, 0.224, 0.225], [1, 1, 3]);
    const scale = tf.scalar(255.);

    status_style.textContent = 'Status: Model loaded! running inference';

    //var bt1 = performance.now();
    //console.log("Call to pre took " + (bt1 - bt0) + " milliseconds.");

    //console.log(tf.memory ());

    var st0 = performance.now();
    const style = model_transformer.predict(style_in);
    var st1 = performance.now();
    style_time = (st1 - st0)
    //console.log("Call to style took " + style_time + " milliseconds.");

    //console.log(tf.memory ());

    //var postt0 = performance.now();

    //const ze = tf.zeros([512, 512, 3]);
    //const on = tf.ones([512, 512, 3]);
    //car=7 person=15
    //const masked = normalised.where(Sem_mask_conc.equal(15), ze);
    //const masked = on.where(Sem_mask_conc.equal(15), ze);
    //const inv_masked = ze.where(Sem_mask_conc.equal(15), normalised);

    const style_out = style.squeeze(0).clipByValue(0, 255).div(scale);

    //var postt1 = performance.now();
    //console.log("Call to post took " + (postt1 - postt0) + " milliseconds.");

    if (output_type == 'style_only') {
      return style_out
    } else {
      const segmentation_img = tf.image.resizeBilinear(img, [segmentation_IMAGE_HEIGHT, segmentation_IMAGE_WIDTH], true)
      const normalised = segmentation_img.div(scale);
      const segmentation_in = segmentation_img.div(scale).sub(mean).div(std).expandDims();

      var sst0 = performance.now();
      const features = model_sem_encoder.predict(segmentation_in);
      const input_feature = tf.image.resizeBilinear(features[4], [64, 64], true)
      const predictions = model_sem_decoder.predict(input_feature);
      var sst1 = performance.now();
      seg_time = (sst1 - sst0)
      //console.log("Call to seg took " + seg_time + " milliseconds.");

      //const out = tf.image.resizeNearestNeighbor(predictions[0],[512,512]).squeeze(0);
      const Sem_mask = tf.image.resizeBilinear(predictions, [segmentation_IMAGE_HEIGHT,
        segmentation_IMAGE_WIDTH
      ], true).squeeze(0).argMax(2); //.expandDims(2);

      const Sem_mask_conc = tf.stack([Sem_mask, Sem_mask, Sem_mask], 2);

      if (output_type == 'masked_style') {
        return tf.image.resizeBilinear(style_out, [segmentation_IMAGE_HEIGHT,
          segmentation_IMAGE_WIDTH
        ], true).where(Sem_mask_conc.equal(15), normalised);
      } else if (output_type == 'masked_style_inverted') {
        return normalised.where(Sem_mask_conc.equal(15),
          tf.image.resizeBilinear(style_out, [segmentation_IMAGE_HEIGHT,
            segmentation_IMAGE_WIDTH
          ], true));
      }
    }

  });

  //console.log(tf.memory ());

  //console.log(`style_out: ${style_out.shape}`);

  t_Width = document.getElementById('inpimg_style').clientWidth
  t_Height = document.getElementById('inpimg_style').clientHeight

  const maskCanvas = document.getElementById('mask');

  status_style.textContent = "Status: Done! inference took " + ((seg_time + style_time).toFixed(1)) + " milliseconds.";

  var tbt0 = performance.now();
  //tf.browser.toPixels(tf.image.resizeBilinear(masked_style_comp,
  //  [IMAGE_HEIGHT,IMAGE_WIDTH]), maskCanvas);
  tf.browser.toPixels(tf.image.resizeBilinear(masked_style_comp,
    [t_Height, t_Width]), maskCanvas);
  //tf.browser.toPixels(style_out_norm, maskCanvas);

  var tbt1 = performance.now();
  //console.log("Call to tb took " + (tbt1 - tbt0) + " milliseconds.");

  var t1 = performance.now();
  //console.log("Call to mask_Demo took " + (t1 - t0) + " milliseconds.");

  masked_style_comp.dispose();
  /*
  normalised.dispose();
  batched.dispose();
  depthPred.dispose();
  depthMask.dispose();
  //tf.disposeVariables();*/
  //console.log("after: ", tf.memory());
  //console.log(tf.memory ());
};

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
  style_IMAGE_HEIGHT = 192 //128
  style_IMAGE_WIDTH = 192 //128
  dropdown_style_qual.textContent = style_low.textContent;
}
style_medium.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  style_IMAGE_HEIGHT = 256
  style_IMAGE_WIDTH = 256
  dropdown_style_qual.textContent = style_medium.textContent;
}
style_high.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  style_IMAGE_HEIGHT = 384 //512
  style_IMAGE_WIDTH = 384 //512
  dropdown_style_qual.textContent = style_high.textContent;
}
style_vhigh.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  style_IMAGE_HEIGHT = 512
  style_IMAGE_WIDTH = 512
  dropdown_style_qual.textContent = style_vhigh.textContent;
}

StyleWarmup();
