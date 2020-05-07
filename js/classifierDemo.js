tf.setBackend('webgl')
//let backend = new tf.webgl.MathBackendWebGL()
tf.ENV.set('WEBGL_CONV_IM2COL', false);

//console.log(tf.ENV.features)
//tf.ENV.set('BEFORE_PAGING_CONSTANT ', 1000);
//tf.setBackend('cpu');
//tf.enableProdMode();

let model_classifier;

var cors_api_url = 'https://cors-anywhere.herokuapp.com/';

const status_classifier = document.getElementById('status_classifier');


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

    const img = tf.browser.fromPixels(imElement).toFloat();
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

ClassiferWarmup();
