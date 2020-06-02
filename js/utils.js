// util functions
// some functions adapted from tfjs pix2pix demo

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}


/**
 * Toggles between the loading UI and the main canvas UI.
 */
function toggleLoadingUI(
  showLoadingUI, loadingDivId = 'loading', mainDivId = 'main') {
  if (showLoadingUI) {
    document.getElementById(loadingDivId).style.display = 'block';
    document.getElementById(mainDivId).style.display = 'none';
  } else {
    document.getElementById(loadingDivId).style.display = 'none';
    document.getElementById(mainDivId).style.display = 'block';
  }
}

function toTuple({
  y,
  x
}) {
  return [y, x];
}

/**
 * Converts an arary of pixel data into an ImageData object
 */
async function renderToCanvas(a, ctx) {
  const [height, width] = a.shape;
  const imageData = new ImageData(width, height);

  const data = await a.data();

  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const k = i * 3;

    imageData.data[j + 0] = data[k + 0];
    imageData.data[j + 1] = data[k + 1];
    imageData.data[j + 2] = data[k + 2];
    imageData.data[j + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

/**
 * Draw an image on a canvas
 */
function renderImageToCanvas(image, size, canvas) {
  canvas.width = size[0];
  canvas.height = size[1];
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0);
}

/**
 * Set the image size to match input element
 */
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
  return [output_WIDTH,output_HEIGHT];
}

// see https://stackoverflow.com/questions/20600800/js-client-side-exif-orientation-rotate-and-mirror-jpeg-images
function file_infer(image, img_in, status, func) {
  //console.log("running file_infer!");
  status.textContent = 'Status: Fetching image...';
  //var tt = performance.now();
  if (window.File && window.FileReader && window.FileList && window.Blob) {
    var reader = new FileReader();
    // Closure to capture the file information.
    reader.addEventListener("load", function(e) {
      const imageData = e.target.result;
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
            //let img_out = document.getElementById('inpimg0');
            img_in.src = canvas.toDataURL();
            //console.log('orientation took: ');
            //console.log(performance.now()-tt);
            img_in.onload = () => func(img_in);
          });
        }
      });
    });
    reader.readAsDataURL(image);
  } else {
    console.log('The File APIs are not fully supported in this browser.');
  }
};

function url_infer(url_in, img_in, status, func){
  //console.log("running url_infer!");
  status.textContent = 'Status: Fetching image...';

  let url = new URL(url_in.value);

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
      //let img = document.getElementById('inpimg0');
      img_in.src = e.target.result;
      //img.height = IMAGE_HEIGHT;
      //img.width = IMAGE_HEIGHT;
      img_in.onload = () => func(img_in);
    };
  };
}

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
