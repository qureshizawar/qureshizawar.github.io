// util functions
// some functions adapted from tfjs pix2pix demo

const cors_api_url = 'https://cors-anywhere.herokuapp.com/';

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

function is_touch_device() {
  return 'ontouchstart' in window // works on most browsers
    ||
    'onmsgesturechange' in window; // works on ie10
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
  return [output_WIDTH, output_HEIGHT];
}

function file_load(image, img_in, status) {
  //console.log("running file_infer!");
  status.textContent = 'Status: Fetching image...';
  //var tt = performance.now();
  if (window.File && window.FileReader && window.FileList && window.Blob) {
    var reader = new FileReader();
    // Closure to capture the file information.
    reader.addEventListener("load", function(e) {
      img_in.src = e.target.result;
      // img_in.onload = () => func(img_in);
    });
    reader.readAsDataURL(image);
  } else {
    console.log('The File APIs are not fully supported in this browser.');
  }
};

function url_load(url_in, img_in, status) {
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
      // Fill the image & call func.
      img_in.src = e.target.result;
      // img_in.onload = () => func(img_in);
    };
  };
}

function file_infer(image, img_in, status, func) {
  //console.log("running file_infer!");
  status.textContent = 'Status: Fetching image...';
  //var tt = performance.now();
  if (window.File && window.FileReader && window.FileList && window.Blob) {

    /*getImageUrl(image).then(url => {
      img_in.src = url;
      img_in.onload = () => func(img_in);
    })*/

    var reader = new FileReader();
    // Closure to capture the file information.
    reader.addEventListener("load", function(e) {
      img_in.src = e.target.result;
      img_in.onload = () => func(img_in);
    });
    reader.readAsDataURL(image);

    // see https://stackoverflow.com/questions/20600800/js-client-side-exif-orientation-rotate-and-mirror-jpeg-images
    /*var reader = new FileReader();
    // Closure to capture the file information.
    reader.addEventListener("load", function(e) {
      const imageData = e.target.result;
      window.loadImage(imageData, function(img, data) {
        if (img.type === "error") {
          console.log("couldn't load image:", img);
        } else {
          //window.EXIF.getData(img, function() {
          //console.log("done!");
          var orientation = data.exif; //exif.get('Orientation') //window.EXIF.getTag(this, "Orientation");
          var canvas = window.loadImage.scale(img, {
            orientation: Number(orientation) || true,
            canvas: true
          });
          //console.log(orientation);
          //document.getElementById("container2").appendChild(canvas);
          // or using jquery $("#container").append(canvas);
          //let img_out = document.getElementById('inpimg0');
          img_in.src = canvas.toDataURL();
          //console.log('orientation took: ');
          //console.log(performance.now()-tt);
          img_in.onload = () => func(img_in);
          //});
        }
      }, {
        meta: true
      });
    });
    reader.readAsDataURL(image);*/
  } else {
    console.log('The File APIs are not fully supported in this browser.');
  }
};

function url_infer(url_in, img_in, status, func) {
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
      // Fill the image & call func.
      img_in.src = e.target.result;
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
  // video.width = videoWidth;
  // video.height = videoHeight;

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: mode == 'rear' ? "environment" : 'user',
      // width: mobile ? undefined : videoWidth,
      // height: mobile ? undefined : videoHeight,
      // width: {ideal : videoWidth},
      // height: {ideal : videoHeight},
      frameRate: { ideal: 30, max: 35 },
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

// Based on: https://stackoverflow.com/a/46814952/283851
// Based on: https://gist.github.com/mindplay-dk/72f47c1a570e870a375bd3dbcb9328fb
/**
 * Create a Base64 Image URL, with rotation applied to compensate for EXIF orientation, if needed.
 *
 * Optionally resize to a smaller maximum width - to improve performance for larger image thumbnails.
 */
function getImageUrl(file, maxWidth) {
  return readOrientation(file).then(function(orientation) {
    return applyRotation(file, orientation || 1, maxWidth || 999999);
  });
}
/**
 * @returns EXIF orientation value (or undefined)
 */
var readOrientation = function(file) {
  return new Promise(function(resolve) {
    var reader = new FileReader();
    reader.onload = function() {
      return resolve((function() {
        var view = new DataView( /** @type {ArrayBuffer} */ (reader.result));
        if (view.getUint16(0, false) != 0xFFD8) {
          return;
        }
        var length = view.byteLength;
        var offset = 2;
        while (offset < length) {
          var marker = view.getUint16(offset, false);
          offset += 2;
          if (marker == 0xFFE1) {
            offset += 2;
            if (view.getUint32(offset, false) != 0x45786966) {
              return;
            }
            offset += 6;
            var little = view.getUint16(offset, false) == 0x4949;
            offset += view.getUint32(offset + 4, little);
            var tags = view.getUint16(offset, little);
            offset += 2;
            for (var i = 0; i < tags; i++) {
              if (view.getUint16(offset + (i * 12), little) == 0x0112) {
                return view.getUint16(offset + (i * 12) + 8, little);
              }
            }
          } else if ((marker & 0xFF00) != 0xFF00) {
            break;
          } else {
            offset += view.getUint16(offset, false);
          }
        }
      })());
    };
    reader.readAsArrayBuffer(file.slice(0, 64 * 1024));
  });
};
/**
 * @returns Base64 Image URL (with rotation applied to compensate for orientation, if any)
 */
var applyRotation = function(file, orientation, maxWidth) {
  return new Promise(function(resolve) {
    var reader = new FileReader();
    reader.onload = function() {
      var url = reader.result;
      var image = new Image();
      image.onload = function() {
        var canvas = document.createElement("canvas");
        var context = canvas.getContext("2d");
        var width = image.width,
          height = image.height;
        var _a = orientation >= 5 && orientation <= 8 ? [height, width] : [width, height],
          outputWidth = _a[0],
          outputHeight = _a[1];
        var scale = outputWidth > maxWidth ? maxWidth / outputWidth : 1;
        width = Math.floor(width * scale);
        height = Math.floor(height * scale);
        // to rotate rectangular image, we need enough space so square canvas is used
        var wh = Math.max(width, height);
        // set proper canvas dimensions before transform & export
        canvas.width = wh;
        canvas.height = wh;
        // for some transformations output image will be aligned to the right or bottom of square canvas
        var rightAligned = false;
        var bottomAligned = false;
        // transform context before drawing image
        switch (orientation) {
          case 2:
            context.transform(-1, 0, 0, 1, wh, 0);
            rightAligned = true;
            break;
          case 3:
            context.transform(-1, 0, 0, -1, wh, wh);
            rightAligned = true;
            bottomAligned = true;
            break;
          case 4:
            context.transform(1, 0, 0, -1, 0, wh);
            bottomAligned = true;
            break;
          case 5:
            context.transform(0, 1, 1, 0, 0, 0);
            break;
          case 6:
            context.transform(0, 1, -1, 0, wh, 0);
            rightAligned = true;
            break;
          case 7:
            context.transform(0, -1, -1, 0, wh, wh);
            rightAligned = true;
            bottomAligned = true;
            break;
          case 8:
            context.transform(0, -1, 1, 0, 0, wh);
            bottomAligned = true;
            break;
          default:
            break;
        }
        // draw image
        context.drawImage(image, 0, 0, width, height);
        // copy rotated image to output dimensions and export it
        var canvas2 = document.createElement("canvas");
        canvas2.width = Math.floor(outputWidth * scale);
        canvas2.height = Math.floor(outputHeight * scale);
        var ctx2 = canvas2.getContext("2d");
        var sx = rightAligned ? canvas.width - canvas2.width : 0;
        var sy = bottomAligned ? canvas.height - canvas2.height : 0;
        ctx2.drawImage(canvas, sx, sy, canvas2.width, canvas2.height, 0, 0, canvas2.width, canvas2.height);
        // export base64
        resolve(canvas2.toDataURL("image/jpeg"));
      };
      image.src = url;
    };
    reader.readAsDataURL(file);
  });
};
