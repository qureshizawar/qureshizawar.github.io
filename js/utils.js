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
