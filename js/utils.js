const color = 'aqua';
const boundingBoxColor = 'red';
const lineWidth = 2;

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
