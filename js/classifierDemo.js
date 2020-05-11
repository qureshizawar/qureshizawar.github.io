tf.setBackend('webgl')
//let backend = new tf.webgl.MathBackendWebGL()
tf.ENV.set('WEBGL_CONV_IM2COL', false);

//console.log(tf.ENV.features)
//tf.ENV.set('BEFORE_PAGING_CONSTANT ', 1000);
//tf.setBackend('cpu');
//tf.enableProdMode();

var videoWidth = 500;
var videoHeight = 600;

const IMAGE_WIDTH = 224;
const IMAGE_HEIGHT = 224;

var mode = 'user' //'user'

//const stats = new Stats();

var cors_api_url = 'https://cors-anywhere.herokuapp.com/';

let model_classifier;
const status_classifier = document.getElementById('status_classifier');

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera(mode) {
  //console.log(navigator.mediaDevices);
  //console.log(navigator.mediaDevices.getUserMedia);
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: mode == 'rear' ? "environment" : 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  //console.log(video.srcObject.getTracks())

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo(mode) {
  const video = await setupCamera(mode);
  video.play();

  return video;
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
/*function setupFPS() {
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  //document.getElementById('main').replaceChild(stats.dom, document.getElementById('fps'));
  document.getElementById('fps').appendChild(stats.dom);
}*/

function classifier_file(image) {
  status_classifier.textContent = 'Status: Fetching image...';
  //var tt = performance.now();
  //console.log(document.getElementById("files0"))
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
            let img_out = document.getElementById('inpimg0');
            img_out.src = canvas.toDataURL();
            //console.log('orientation took: ');
            //console.log(performance.now()-tt);
            img_out.onload = () => classifier_Demo(img_out);
          });
        }
      });
    });
    reader.readAsDataURL(image);
  } else {
    console.log('The File APIs are not fully supported in this browser.');
  }
};

document.getElementById("files0").addEventListener("change", function(evt) {
  classifier_file(evt.target.files[0]);
});

document.getElementById("classifier_files_btn").addEventListener("click", function(evt) {
  file = document.getElementById("files0").files[0];
  if (file == null) {
    status_classifier.textContent = 'Status: File not found';
  } else {
    classifier_file(file);
  }
});

document.getElementById('btn0').onclick = function() {

  status_classifier.textContent = 'Status: Fetching image...';

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
      //img.height = IMAGE_HEIGHT;
      //img.width = IMAGE_HEIGHT;
      img.onload = () => classifier_Demo(img);
    };
  };
}

const ClassiferWarmup = async () => {

  status_classifier.textContent = 'Status: Loading...';
  model_classifier = await tf.loadGraphModel('/assets/tfjs_model/model.json');

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  //model_classifier.predict(tf.zeros([1, IMAGE_HEIGHT, IMAGE_HEIGHT, 3])).dispose();

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

  //var t0 = performance.now();

  status_classifier.textContent = 'Status: Loading image into model...';

  const out = tf.tidy(() => {

    const img_tensor = tf.browser.fromPixels(imElement).toFloat();
    const img = img_tensor.resizeBilinear([IMAGE_HEIGHT, IMAGE_WIDTH]);
    const scale = tf.scalar(255.);
    const mean = tf.tensor3d([0.485, 0.456, 0.406], [1, 1, 3]);
    const std = tf.tensor3d([0.229, 0.224, 0.225], [1, 1, 3]);
    const normalised = img.div(scale).sub(mean).div(std);
    status_classifier.textContent = 'Status: Model loaded! running inference';
    it0 = performance.now();
    const batched = normalised.transpose([0, 1, 2]).expandDims();

    return model_classifier.predict(batched).arraySync();
  });
  it1 = performance.now();
  var output = [];

  //out = predictions.arraySync();
  output.push(["bus", out[0][0]]);
  output.push(["car", out[0][1]]);
  output.push(["pickup", out[0][2]]);
  output.push(["truck", out[0][3]]);
  output.push(["van", out[0][4]]);

  //console.log(output);
  // done to sort vals as numbers instead of strings
  output.sort(function(a, b) {
    return b[1] - a[1]
  });
  document.getElementById("classifier_out1").innerHTML = output[0][0] + ": " + (output[0][1] * 100).toFixed(2) + "%";
  document.getElementById("classifier_out2").innerHTML = output[1][0] + ": " + (output[1][1] * 100).toFixed(2) + "%";
  document.getElementById("classifier_out3").innerHTML = output[2][0] + ": " + (output[2][1] * 100).toFixed(2) + "%";

  //var t1 = performance.now();

  status_classifier.textContent = "Status: Done! inference took " + ((it1 - it0).toFixed(1)) + " milliseconds.";

  //console.log("before: ", tf.memory());
  //tf.disposeVariables();
  //console.log("after: ", tf.memory());
  //console.log("Call to classifier_Demo took " + (t1 - t0) + " milliseconds.");
};

let request;

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video) {
  console.log("running detectPoseInRealTime!")
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');

  // since images are being fed from a webcam, we want to feed in the
  // original image and then just flip the keypoints' x coordinates. If instead
  // we flip the image, then correcting left-right keypoint pairs requires a
  // permutation on all the keypoints.
  const flipHorizontal = mode == 'rear' ? false : true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function poseDetectionFrame() {

    console.log("running poseDetectionFrame!")

    // Begin monitoring code for frames per second
    //stats.begin();

    ctx.clearRect(0, 0, videoWidth, videoHeight);
    //console.log(video.onloadeddata)
    video.onloadeddata = () => {
      camloaded = true;
      //console.log(video.srcObject)
    }

    if (camloaded) {
      await classifier_Demo(video);

      ctx.save();
      if (flipHorizontal) {
        ctx.scale(-1, 1);
        ctx.translate(-videoWidth, 0);
      }
      /*else{
        ctx.scale(1, 1);
      }*/
      ctx.scale(1, 1);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }

    /*if (document.getElementById("show_fps").checked) {
      setupFPS();
      //document.getElementById('main').replaceChild(stats.dom, document.getElementById('fps'));
    }*/

    //console.log(video)

    // End monitoring code for frames per second
    //stats.end();

    request = requestAnimationFrame(poseDetectionFrame);

  }

  /*if (detect) {
    prom = new Promise((resolve) => {
      poseDetectionFrame();
      resolve();
    });*/
  poseDetectionFrame();
}


let video;

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
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


  //console.log(video.srcObject.getTracks())

  //setupGui([], net);
  //setupFPS();
  detectPoseInRealTime(video);
}

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
var camloaded = false;

function showfps() {
  var show_fps = document.getElementById("show_fps");
  var fpsv = document.getElementById("fps");
  if (show_fps.checked == true) {
    fpsv.style.display = "block";
  } else {
    fpsv.style.display = "none";
  }
}

let prom;

function setmode() {
  if (document.getElementById("webcamc").checked == true) {
    if (document.getElementById("camMode").checked == true) {
      mode = 'rear';

      video.srcObject.getTracks().forEach(function(track) {
        track.stop();
      });

      cancelAnimationFrame(request);

      console.log("camMode");

      //video = loadVideo(mode);

      /*prom.finally(() => {
        console.log("promise complete!")
        bindPage()
        //webc()
      });*/
      bindPage()
    } else {
      mode = 'user';
      video.srcObject.getTracks().forEach(function(track) {
        track.stop();
      });

      cancelAnimationFrame(request);

      console.log("camMode");
      bindPage()
    }
  }
}


function webc() {
  var imagecheckBox = document.getElementById("imagec");
  var videocheckBox = document.getElementById("webcamc");
  var maini = document.getElementById("mainImage");
  var mainv = document.getElementById("mainVideo");

  if (videocheckBox.checked == true) {

    imagecheckBox.checked = false;
    //var style = window.getComputedStyle(inpimg);
    /*console.log(style);
    console.log(style.getPropertyValue('width'));
    console.log(style.getPropertyValue('height'));*/

    t_Width = document.getElementById("main").clientWidth
    t_Height = 300; //inpimg.clientHeight
    maini.style.display = "none";
    mainv.style.display = "block";

    apectWidth = (4 / 3) * t_Height
    videoWidth = t_Width > apectWidth ? apectWidth : t_Width //t_Width//860;//style.getPropertyValue('width');
    videoHeight = t_Height //300;//style.getPropertyValue('height');

    /*console.log(videoWidth);
    console.log(videoHeight);*/

    camloaded = false;
    //let video;

    bindPage();
    /*prom = new Promise((resolve) => {
      bindPage();
      resolve();
    });*/
    //text.style.display = "block";
  } else {
    //delete video;
    console.log(video.srcObject.getTracks())
    /*prom.finally( ()=> {
      console.log("promise complete!")
    });*/

    cancelAnimationFrame(request);
    video.srcObject.getTracks().forEach(function(track) {
      track.stop();
    });
    console.log(video.srcObject.getTracks())
    mainv.style.display = "none";
  }
}

function imagec() {
  var imagecheckBox = document.getElementById("imagec");
  var videocheckBox = document.getElementById("webcamc");
  var maini = document.getElementById("mainImage");
  var inpimg = document.getElementById("inpimg0");
  var mainv = document.getElementById("mainVideo");

  if (imagecheckBox.checked == true) {
    if (videocheckBox.checked == true) {

      videocheckBox.checked = false;
      cancelAnimationFrame(request);
      video.srcObject.getTracks().forEach(function(track) {
        track.stop();
      });
    }
    maini.style.display = "block";
    mainv.style.display = "none";
    inpimg.src = "/assets/demo_images/box_6109.jpg";
    /*var style = window.getComputedStyle(inpimg);
    console.log(style);
    console.log(style.getPropertyValue('width'));*/
    ClassiferWarmup();
  } else {
    maini.style.display = "none";
  }
}

ClassiferWarmup();
