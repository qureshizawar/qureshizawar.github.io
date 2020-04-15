import { OrbitControls } from './OrbitControls.js';

window.createPointCloud = function createPointCloud(xx, yy, depth_array, img_array){

    var controls;

    function createTextCanvas(text, color, font, size) {
        size = size || 16;
        var canvas = document.createElement('canvas');//document.getElementById('3d_depth');//document.createElement('canvas');
        var ctx = canvas.getContext('2d');
        var fontStr = (size + 'px ') + (font || 'Arial');
        ctx.font = fontStr;
        var w = ctx.measureText(text).width;
        var h = Math.ceil(size);
        canvas.width = w;
        canvas.height = h;
        canvas.id = "depth_canvas"
        ctx.font = fontStr;
        ctx.fillStyle = color || 'black';
        ctx.fillText(text, 0, Math.ceil(size * 0.8));
        return canvas;
    }

    function createText2D(text, color, font, size, segW, segH) {
        var canvas = createTextCanvas(text, color, font, size);
        var plane = new THREE.PlaneGeometry(canvas.width, canvas.height, segW, segH);
        var tex = new THREE.Texture(canvas);
        tex.needsUpdate = true;
        var planeMat = new THREE.MeshBasicMaterial({
            map: tex,
            color: 0xffffff,
            transparent: true
        });
        var mesh = new THREE.Mesh(plane, planeMat);
        mesh.scale.set(0.5, 0.5, 0.5);
        mesh.doubleSided = true;
        return mesh;
    }

    // from http://stackoverflow.com/questions/5623838/rgb-to-hex-and-hex-to-rgb
    function hexToRgb(hex) { //TODO rewrite with vector output
        var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }
    function componentToHex(c) {
      var hex = c.toString(16);
      return hex.length == 1 ? "0" + hex : hex;
    }
    function rgbToHex(r, g, b) {
      return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
    }

    var renderer = new THREE.WebGLRenderer({
        antialias: true
    });
    var w = img_array[0].length;//400;
    var h = img_array.length;//300;
    renderer.setSize(w, h);
    renderer.domElement.id = "depth_canvas"
    //console.log(renderer.domElement.attributes)
    //document.body.appendChild(renderer.domElement);

    var depth_node = document.getElementById('depth_sec');
    //console.log(depth_node.children)
    var depth_canvas_node = document.getElementById('depth_canvas');

    renderer.setClearColorHex(0xEEEEEE, 1.0);

    function normFOV(h,w){
        return (1/(1+Math.pow(Math.E, 5-4*(w/h)))) + 0.5
    }
    //console.log(w/h)
    //console.log(normFOV(h,w))

    var camera = new THREE.PerspectiveCamera(45, w / h, 1, 10000);
    camera.position.z = 200;
    camera.position.x = 100;
    camera.position.y = 100;

    var scene = new THREE.Scene();

    var scatterPlot = new THREE.Object3D();
    scene.add(scatterPlot);

    scatterPlot.rotation.y = 0;

    function v(x, y, z) {
        return new THREE.Vector3(x, y, z);
    }

    var unfiltered = [],
        lowPass = [],
        highPass = [];

    var format = d3.format("+.3f");

    for (var i=0; i<xx.length; i++){
        for (var j=0; j<xx[0].length; j++){
            unfiltered[i*xx[0].length +j] = {
                x: yy[i][j],//xx[i][j],//+d.x,
                y: -xx[i][j],//yy[i][j],//+d.y,
                z: depth_array[i][j],//+d.z,
                r: img_array[i][j][0],//+d.r,
                g: img_array[i][j][1],//+d.g,
                b: img_array[i][j][2]//+d.b
            };
        }
    }

    var xExent = d3.extent(unfiltered, function (d) {return d.x; }),
        yExent = d3.extent(unfiltered, function (d) {return d.y; }),
        zExent = d3.extent(unfiltered, function (d) {return d.z; });

    var vpts = {
        xMax: xExent[1],
        xCen: (xExent[1] + xExent[0]) / 2,
        xMin: xExent[0],
        yMax: yExent[1],
        yCen: (yExent[1] + yExent[0]) / 2,
        yMin: yExent[0],
        zMax: zExent[1],
        zCen: (zExent[1] + zExent[0]) / 2,
        zMin: zExent[0]
    }

    var colour = d3.scale.category20c();

    var xScale = d3.scale.linear()
                  .domain(xExent)
                  .range([-w/3,w/3]);
    var yScale = d3.scale.linear()
                  .domain(yExent)
                  .range([-h/3,h/3]);                  
    var zScale = d3.scale.linear()
                  .domain(zExent)
                  .range([-75,75]);

    var lineGeo = new THREE.Geometry();
    /*lineGeo.vertices.push(
        v(xScale(vpts.xMin), yScale(vpts.yCen), zScale(vpts.zCen)), v(xScale(vpts.xMax), yScale(vpts.yCen), zScale(vpts.zCen)),
        v(xScale(vpts.xCen), yScale(vpts.yMin), zScale(vpts.zCen)), v(xScale(vpts.xCen), yScale(vpts.yMax), zScale(vpts.zCen)),
        v(xScale(vpts.xCen), yScale(vpts.yCen), zScale(vpts.zMax)), v(xScale(vpts.xCen), yScale(vpts.yCen), zScale(vpts.zMin)),

        v(xScale(vpts.xMin), yScale(vpts.yMax), zScale(vpts.zMin)), v(xScale(vpts.xMax), yScale(vpts.yMax), zScale(vpts.zMin)),
        v(xScale(vpts.xMin), yScale(vpts.yMin), zScale(vpts.zMin)), v(xScale(vpts.xMax), yScale(vpts.yMin), zScale(vpts.zMin)),
        v(xScale(vpts.xMin), yScale(vpts.yMax), zScale(vpts.zMax)), v(xScale(vpts.xMax), yScale(vpts.yMax), zScale(vpts.zMax)),
        v(xScale(vpts.xMin), yScale(vpts.yMin), zScale(vpts.zMax)), v(xScale(vpts.xMax), yScale(vpts.yMin), zScale(vpts.zMax)),

        v(xScale(vpts.xMin), yScale(vpts.yCen), zScale(vpts.zMax)), v(xScale(vpts.xMax), yScale(vpts.yCen), zScale(vpts.zMax)),
        v(xScale(vpts.xMin), yScale(vpts.yCen), zScale(vpts.zMin)), v(xScale(vpts.xMax), yScale(vpts.yCen), zScale(vpts.zMin)),
        v(xScale(vpts.xMin), yScale(vpts.yMax), zScale(vpts.zCen)), v(xScale(vpts.xMax), yScale(vpts.yMax), zScale(vpts.zCen)),
        v(xScale(vpts.xMin), yScale(vpts.yMin), zScale(vpts.zCen)), v(xScale(vpts.xMax), yScale(vpts.yMin), zScale(vpts.zCen)),

        v(xScale(vpts.xMax), yScale(vpts.yMin), zScale(vpts.zMin)), v(xScale(vpts.xMax), yScale(vpts.yMax), zScale(vpts.zMin)),
        v(xScale(vpts.xMin), yScale(vpts.yMin), zScale(vpts.zMin)), v(xScale(vpts.xMin), yScale(vpts.yMax), zScale(vpts.zMin)),
        v(xScale(vpts.xMax), yScale(vpts.yMin), zScale(vpts.zMax)), v(xScale(vpts.xMax), yScale(vpts.yMax), zScale(vpts.zMax)),
        v(xScale(vpts.xMin), yScale(vpts.yMin), zScale(vpts.zMax)), v(xScale(vpts.xMin), yScale(vpts.yMax), zScale(vpts.zMax)),

        v(xScale(vpts.xCen), yScale(vpts.yMin), zScale(vpts.zMax)), v(xScale(vpts.xCen), yScale(vpts.yMax), zScale(vpts.zMax)),
        v(xScale(vpts.xCen), yScale(vpts.yMin), zScale(vpts.zMin)), v(xScale(vpts.xCen), yScale(vpts.yMax), zScale(vpts.zMin)),
        v(xScale(vpts.xMax), yScale(vpts.yMin), zScale(vpts.zCen)), v(xScale(vpts.xMax), yScale(vpts.yMax), zScale(vpts.zCen)),
        v(xScale(vpts.xMin), yScale(vpts.yMin), zScale(vpts.zCen)), v(xScale(vpts.xMin), yScale(vpts.yMax), zScale(vpts.zCen)),

        v(xScale(vpts.xMax), yScale(vpts.yMax), zScale(vpts.zMin)), v(xScale(vpts.xMax), yScale(vpts.yMax), zScale(vpts.zMax)),
        v(xScale(vpts.xMax), yScale(vpts.yMin), zScale(vpts.zMin)), v(xScale(vpts.xMax), yScale(vpts.yMin), zScale(vpts.zMax)),
        v(xScale(vpts.xMin), yScale(vpts.yMax), zScale(vpts.zMin)), v(xScale(vpts.xMin), yScale(vpts.yMax), zScale(vpts.zMax)),
        v(xScale(vpts.xMin), yScale(vpts.yMin), zScale(vpts.zMin)), v(xScale(vpts.xMin), yScale(vpts.yMin), zScale(vpts.zMax)),

        v(xScale(vpts.xMin), yScale(vpts.yCen), zScale(vpts.zMin)), v(xScale(vpts.xMin), yScale(vpts.yCen), zScale(vpts.zMax)),
        v(xScale(vpts.xMax), yScale(vpts.yCen), zScale(vpts.zMin)), v(xScale(vpts.xMax), yScale(vpts.yCen), zScale(vpts.zMax)),
        v(xScale(vpts.xCen), yScale(vpts.yMax), zScale(vpts.zMin)), v(xScale(vpts.xCen), yScale(vpts.yMax), zScale(vpts.zMin)),
        v(xScale(vpts.xCen), yScale(vpts.yMin), zScale(vpts.zMin)), v(xScale(vpts.xCen), yScale(vpts.yMin), zScale(vpts.zMax))

    );*/
    var lineMat = new THREE.LineBasicMaterial({
        color: 0x000000,
        lineWidth: 1
    });
    var line = new THREE.Line(lineGeo, lineMat);
    line.type = THREE.Lines;
    scatterPlot.add(line);

    var titleX = createText2D('-X');
    titleX.position.x = xScale(vpts.xMin) - 12,
    titleX.position.y = 5;
    scatterPlot.add(titleX);

    var valueX = createText2D(format(xExent[0]));
    valueX.position.x = xScale(vpts.xMin) - 12,
    valueX.position.y = -5;
    scatterPlot.add(valueX);

    var titleX = createText2D('X');
    titleX.position.x = xScale(vpts.xMax) + 12;
    titleX.position.y = 5;
    scatterPlot.add(titleX);

    var valueX = createText2D(format(xExent[1]));
    valueX.position.x = xScale(vpts.xMax) + 12,
    valueX.position.y = -5;
    scatterPlot.add(valueX);

    var titleY = createText2D('-Y');
    titleY.position.y = yScale(vpts.yMin) - 5;
    scatterPlot.add(titleY);

    var valueY = createText2D(format(yExent[0]));
    valueY.position.y = yScale(vpts.yMin) - 15,
    scatterPlot.add(valueY);

    var titleY = createText2D('Y');
    titleY.position.y = yScale(vpts.yMax) + 15;
    scatterPlot.add(titleY);

    var valueY = createText2D(format(yExent[1]));
    valueY.position.y = yScale(vpts.yMax) + 5,
    scatterPlot.add(valueY);

    var titleZ = createText2D('-Z ' + format(zExent[0]));
    titleZ.position.z = zScale(vpts.zMin) + 2;
    scatterPlot.add(titleZ);

    var titleZ = createText2D('Z ' + format(zExent[1]));
    titleZ.position.z = zScale(vpts.zMax) + 2;
    scatterPlot.add(titleZ);

    var mat = new THREE.ParticleBasicMaterial({
        vertexColors: true,
        //color: 0x888888,
        size: 1
    });

    var pointCount = unfiltered.length;
    /*console.log('point count: ')
    console.log(pointCount)
    console.log(unfiltered[0].r, unfiltered[0].g, unfiltered[0].b)*/
    //console.log(hexToRgb(colour(0)).r / 255, hexToRgb(colour(0)).g / 255, hexToRgb(colour(0)).b / 255)
    var pointGeo = new THREE.Geometry();
    for (var i = 0; i < pointCount; i ++) {
        var x = xScale(unfiltered[i].x);
        var y = yScale(unfiltered[i].y);
        var z = zScale(unfiltered[i].z);

        pointGeo.vertices.push(new THREE.Vector3(x, y, z));
        //console.log(pointGeo.vertices);
        //pointGeo.vertices[i].angle = Math.atan2(z, x);
        //pointGeo.vertices[i].radius = Math.sqrt(x * x + z * z);
        //pointGeo.vertices[i].speed = (z / 100) * (x / 100);
        /*pointGeo.colors.push(new THREE.Color().setRGB(
          hexToRgb(colour(i)).r / 255, hexToRgb(colour(i)).g / 255, hexToRgb(colour(i)).b / 255 
        ));*/
        pointGeo.colors.push(new THREE.Color().setRGB(unfiltered[i].r/255, unfiltered[i].g/255, unfiltered[i].b/255
        ));

    }

    var points = new THREE.ParticleSystem(pointGeo, mat);
    scatterPlot.add(points);
    //})

    //document.body.appendChild( renderer.domElement );
    //document.getElementById('depth_sec').appendChild( renderer.domElement );
    depth_node.replaceChild(renderer.domElement,depth_canvas_node);
    // controls

    controls = new OrbitControls( camera, renderer.domElement );

    //controls.addEventListener( 'change', render ); // call this only in static scenes (i.e., if there is no animation loop)

    controls.enableDamping = false; // an animation loop is required when either damping or auto-rotation are enabled
    //controls.dampingFactor = 0.05;

    controls.screenSpacePanning = false;

    controls.minDistance = 100;
    controls.maxDistance = 500;

    controls.maxPolarAngle = Math.PI / 2;

    /*renderer.render(scene, camera);
    var paused = false;
    var last = new Date().getTime();
    var down = false;
    var sx = 0,
        sy = 0;
        
    window.onmousedown = function(ev) {
        down = true;
        sx = ev.clientX;
        sy = ev.clientY;
    };
    window.onmouseup = function() {
        down = false;
    };
    window.onmousemove = function(ev) {
        if (down) {
            var dx = ev.clientX - sx;
            var dy = ev.clientY - sy;
            scatterPlot.rotation.y += dx * 0.01;
            camera.position.y += dy;
            sx += dx;
            sy += dy;
        }
    }

    window.ontouchstart = function(ev) {
        console.log('touched!')
        down = true;
        sx = ev.clientX;
        sy = ev.clientY;
    };
    window.ontouchend = function() {
        console.log('untouched!')
        down = false;
    };
    window.ontouchcancel = function() {
        down = false;
    };
    window.ontouchmove = function(ev) {
        if (down) {
            var dx = ev.clientX - sx;
            var dy = ev.clientY - sy;
            scatterPlot.rotation.y += dx * 0.01;
            camera.position.y += dy;
            sx += dx;
            sy += dy;
        }
    }
    var animating = true;//false;
    /*window.ondblclick = function() {
        animating = !animating;
    };*/

    /*function animate(t) {
        if (!paused) {
            last = t;
            if (animating) {
                var v = pointGeo.vertices;
                for (var i = 0; i < v.length; i++) {
                    var u = v[i];
                    //console.log(u)
                    u.angle += u.speed * 0.01;
                    u.x = Math.cos(u.angle) * u.radius;
                    u.z = Math.sin(u.angle) * u.radius;
                }
                pointGeo.__dirtyVertices = true;
            }
            renderer.clear();
            camera.lookAt(scene.position);
            renderer.render(scene, camera);
        }
        window.requestAnimationFrame(animate, renderer.domElement);
    };
    animate(new Date().getTime());
    onmessage = function(ev) {
        paused = (ev.data == 'pause');
    };*/

    //})

    animate();

    function animate() {

        requestAnimationFrame( animate );

        //controls.update(); // only required if controls.enableDamping = true, or if controls.autoRotate = true

        render();

    }

    function render() {

        renderer.render( scene, camera );

    }
}

//export { createPointCloud };