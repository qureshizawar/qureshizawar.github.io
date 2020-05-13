tf.setBackend('webgl')
//let backend = new tf.webgl.MathBackendWebGL()
tf.ENV.set('WEBGL_CONV_IM2COL', false);
tf.ENV.set('WEBGL_PACK', false); // This needs to be done otherwise things run very slow v1.0.4
tf.webgl.forceHalfFloat()

//console.log(tf.ENV.features)
//tf.ENV.set('BEFORE_PAGING_CONSTANT ', 1000);
//tf.setBackend('cpu');
//tf.enableProdMode();

let model_depth_encoder;
let model_depth_decoder;

var Depth_IMAGE_HEIGHT = 192
var Depth_IMAGE_WIDTH = 320

var output_HEIGHT = 300;
var output_WIDTH = 400;

var cors_api_url = 'https://cors-anywhere.herokuapp.com/';
const status_depth = document.getElementById('status_depth');

const dropdown_depth_qual = document.getElementById('dropdown_depth_qual');
const depth_low = document.getElementById('depth_low')
const depth_medium = document.getElementById('depth_medium')
const depth_high = document.getElementById('depth_high')

function is_touch_device() {
  return 'ontouchstart' in window // works on most browsers
    ||
    'onmsgesturechange' in window; // works on ie10
}
//console.log(screen.width)
//console.log((window.matchMedia('(max-device-width: 960px)').matches))
if (!(is_touch_device()) && !(window.matchMedia('(max-device-width: 960px)').matches)) {
  Depth_IMAGE_HEIGHT = 192
  Depth_IMAGE_WIDTH = 640
  dropdown_depth_qual.textContent = depth_high.textContent;
  //console.log("Desktop detected!")
}

function set_static_output_size(element) {
  /*console.log("clientHeight", element.clientHeight)
  console.log("clientWidth", element.clientWidth)
  console.log("naturalHeight", element.naturalHeight)
  console.log("naturalWidth", element.naturalWidth)*/
  //too small
  if (element.clientWidth > element.naturalWidth && element.clientHeight > element.naturalHeight) {
    output_HEIGHT = element.clientHeight
    output_WIDTH = Math.round((element.naturalWidth / element.naturalHeight) * output_HEIGHT);
    if (output_WIDTH > element.clientWidth) {
      output_WIDTH = element.clientWidth
      output_HEIGHT = Math.round((element.naturalHeight / element.naturalWidth) * output_WIDTH);
    }
    //too big
  } else if (element.clientWidth < element.naturalWidth && element.clientHeight < element.naturalHeight) {
    output_WIDTH = element.clientWidth
    output_HEIGHT = Math.round((element.naturalHeight / element.naturalWidth) * output_WIDTH);
    if (output_HEIGHT > element.clientHeight) {
      output_HEIGHT = element.clientHeight
      output_WIDTH = Math.round((element.naturalWidth / element.naturalHeight) * output_HEIGHT);
    }
    //too long
  } else if (element.clientWidth < element.naturalWidth) {
    output_WIDTH = element.clientWidth
    output_HEIGHT = Math.round((element.naturalHeight / element.naturalWidth) * output_WIDTH);
  } //too tall
  else {
    output_HEIGHT = element.clientHeight
    output_WIDTH = Math.round((element.naturalWidth / element.naturalHeight) * output_HEIGHT);
  }
  /*output_HEIGHT = element.clientHeight
  output_WIDTH = element.clientWidth*/
  /*console.log(output_HEIGHT);
  console.log(output_WIDTH);*/
}

function depth_file(image) {
  status_depth.textContent = 'Status: Fetching image...';
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
            let img_out = document.getElementById('inpimg');
            img_out.src = canvas.toDataURL();
            //console.log('orientation took: ');
            //console.log(performance.now()-tt);
            img_out.onload = () => {
              set_static_output_size(img_out);
              Depth_Demo(img_out);
            }
          });
        }
      });
    });
    reader.readAsDataURL(image);
  } else {
    console.log('The File APIs are not fully supported in this browser.');
  }
};

document.getElementById("files").addEventListener("change", function(evt) {
  depth_file(evt.target.files[0]);
});
document.getElementById("depth_files_btn").addEventListener("click", function(evt) {
  file = document.getElementById("files").files[0];
  if (file == null) {
    status_depth.textContent = 'Status: File not found';
  } else {
    depth_file(file);
  }
});

document.getElementById('btn').onclick = function() {
  status_depth.textContent = 'Status: Fetching image...';
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
      //img.height = IMAGE_HEIGHT;
      //img.width = IMAGE_WIDTH;
      img.onload = () => {
        set_static_output_size(img);
        Depth_Demo(img);
      }
    };
  };
}

function custom_mgrid(rows, cols) {
  a = tf.tile(tf.range(0, rows).reshape([rows, 1]), [1, cols]) // cols
  b = tf.tile(tf.range(0, cols).expandDims(0), [rows, 1]) // rows
  return [a, b];
}

const DepthWarmup = async () => {

  status_depth.textContent = 'Status: Loading...';
  model_depth_encoder = await tf.loadLayersModel('/assets/tfjs_encoder_quant/model.json');
  model_depth_decoder = await tf.loadLayersModel('/assets/tfjs_decoder_quant/model.json');

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

  document.getElementById('file-container').style.display = '';
};


const Depth_Demo = async (imElement) => {

  /*var canvas = document.createElement('canvas');
  canvas.height = output_HEIGHT//imElement.height;
  canvas.width = output_WIDTH//imElement.width;
  var tmpctx = canvas.getContext('2d');
  tmpctx.drawImage( imElement, 0, 0, output_WIDTH, output_HEIGHT );*/

  //var t0 = performance.now();
  var depth_time = 0

  status_depth.textContent = 'Status: Loading image into model...';

  in_img = tf.browser.fromPixels(imElement).toFloat();

  const depthMask = tf.tidy(() => {

    const img = tf.image.resizeBilinear(tf.browser.fromPixels(imElement).toFloat(),
      [Depth_IMAGE_HEIGHT, Depth_IMAGE_WIDTH]);
    //console.log(img);
    const scale = tf.scalar(255.);
    const normalised = img.div(scale);

    status_depth.textContent = 'Status: Model loaded! running inference';
    const batched = normalised.transpose([2, 0, 1]).expandDims();

    //const predictions = model.predict(batched);

    var it0 = performance.now();
    const features = model_depth_encoder.predict(batched);
    const predictions = model_depth_decoder.predict(features);
    var it1 = performance.now();
    depth_time = (it1 - it0);

    const depthPred = predictions[3].squeeze(0).transpose([1, 2, 0]);

    return depthPred.sub(depthPred.min()).divNoNan(depthPred.max().sub(depthPred.min()));

  });

  const depthCanvas = document.getElementById('depth');
  depthCanvas.width = output_WIDTH;
  depthCanvas.height = output_HEIGHT;
  /*t_Width = document.getElementById('inpimg').clientWidth
  t_Height = document.getElementById('inpimg').clientHeight*/

  //var t1 = performance.now();
  //console.log("Call to depth took " + (t1 - t0) + " milliseconds.");

  //await tf.browser.toPixels(tf.image.resizeBilinear(depthMask,
  //  [IMAGE_HEIGHT,IMAGE_WIDTH]), depthCanvas);

  const depthMask_resized = tf.image.resizeBilinear(depthMask, [output_HEIGHT, output_WIDTH])

  //img_array = await tmpctx.getImageData(0,0,output_WIDTH,output_HEIGHT);//in_img.arraySync()
  img_array = tf.image.resizeBilinear(in_img, [output_HEIGHT, output_WIDTH]).arraySync()
  depth_array = depthMask_resized.arraySync()

  //const points = GenPointCloud(depthMask_resized)

  await tf.browser.toPixels(depthMask_resized, depthCanvas);

  status_depth.textContent = "Status: Done! inference took " + (depth_time.toFixed(1)) + " milliseconds.";
  //console.log("before: ", tf.memory());

  //tf.disposeVariables();
  //console.log("after: ", tf.memory());


  const xy = custom_mgrid(img_array.length, img_array[0].length)
  createPointCloud(xy[0].arraySync(), xy[1].arraySync(), depth_array, img_array);
};

/*var coll = document.getElementsByClassName("collapsible_tf");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("activeb");
    var content_tf = this.nextElementSibling;
    if (content_tf.style.display === "block") {
      content_tf.style.display = "none";
    } else {
      content_tf.style.display = "block";
      if (this.title == "Depth") {
        DepthWarmup();
      }
    }
  });
}*/


depth_low.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  Depth_IMAGE_HEIGHT = 96;
  Depth_IMAGE_WIDTH = 320;
  dropdown_depth_qual.textContent = depth_low.textContent;
}
depth_medium.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  Depth_IMAGE_HEIGHT = 192
  Depth_IMAGE_WIDTH = 320
  dropdown_depth_qual.textContent = depth_medium.textContent;
}
depth_high.onclick = function() {
  event.preventDefault();
  //console.log("btn pressed!")
  Depth_IMAGE_HEIGHT = 192
  Depth_IMAGE_WIDTH = 640
  dropdown_depth_qual.textContent = depth_high.textContent;
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

DepthWarmup();
