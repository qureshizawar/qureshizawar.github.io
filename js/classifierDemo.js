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

const mobile = isMobile();

//const stats = new Stats();

var cors_api_url = 'https://cors-anywhere.herokuapp.com/';

let model_classifier;
const status_classifier = document.getElementById('status_classifier');

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera(mode) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: mode == 'rear' ? "environment" : 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

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

  status_classifier.textContent = "Status: Done! inference took " + ((it1 - it0).toFixed(1)) + " ms.";

  //console.log("before: ", tf.memory());
  //tf.disposeVariables();
  //console.log("after: ", tf.memory());
  //console.log("Call to classifier_Demo took " + (t1 - t0) + " milliseconds.");
};

let request;

/**
 * Feeds an image to network to do inference - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectInRealTime(video) {
  //console.log("running detectInRealTime!")
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');

  const flipHorizontal = mode == 'rear' ? false : true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function DetectionFrame() {

    //console.log("running DetectionFrame!")

    // Begin monitoring code for frames per second
    //stats.begin();

    ctx.clearRect(0, 0, videoWidth, videoHeight);
    video.onloadeddata = () => {
      camloaded = true;
    }

    if (camloaded) {
      await classifier_Demo(video);

      ctx.save();
      if (flipHorizontal) {
        ctx.scale(-1, 1);
        ctx.translate(-videoWidth, 0);
      }
      /*else{
        ctx.scale(1, 1);*/
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }

    /*if (document.getElementById("show_fps").checked) {
      setupFPS();
      //document.getElementById('main').replaceChild(stats.dom, document.getElementById('fps'));
    }*/

    // End monitoring code for frames per second
    //stats.end();

    request = requestAnimationFrame(DetectionFrame);

  }

  DetectionFrame();
}


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

  //setupFPS();
  detectInRealTime(video);
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
    t_Height = 300;
    maini.style.display = "none";
    document.getElementById("main").style.display = "block";
    mainv.style.display = "block";

    apectWidth = (4 / 3) * t_Height
    videoWidth = apectWidth > t_Width ? t_Width : apectWidth;
    videoHeight = t_Height;

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
    ctx.clearRect(0, 0, videoWidth, videoHeight);
    //console.log(video.srcObject.getTracks())
    mainv.style.display = "none";
    document.getElementById("main").style.display = "none";
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
    inpimg.src = "/assets/demo_images/box_6109.jpg";
    ClassiferWarmup();
  } else {
    maini.style.display = "none";
    document.getElementById("main").style.display = "none";
  }
}

ClassiferWarmup();
