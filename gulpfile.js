var gulp = require('gulp');
var plumber = require('gulp-plumber');
const uglify = require('gulp-uglify');
const sass = require('gulp-sass');
const wait = require('gulp-wait');
const babel = require('gulp-babel');;
const rename = require('gulp-rename');
const terser = require('gulp-terser');

gulp.task('scripts', function() {
    return gulp.src('./js/scripts.js')
        .pipe(plumber(plumber({
            errorHandler: function (err) {
                console.log(err);
                this.emit('end');
            }
        })))
        .pipe(babel({
          presets: [['@babel/env', {modules:false}]]
        }))
        .pipe(uglify({
            output: {
                comments: '/^!/'
            }
        }))
        .pipe(rename({extname: '.min.js'}))
        .pipe(gulp.dest('./js'));
});

gulp.task('classifierDemo', function() {
    return gulp.src('./js/classifierDemo.js')
        .pipe(terser())
        .pipe(rename({extname: '.min.js'}))
        .pipe(gulp.dest('./js'));
});

gulp.task('depthDemo', function() {
    return gulp.src('./js/depthDemo.js')
        .pipe(terser())
        .pipe(rename({extname: '.min.js'}))
        .pipe(gulp.dest('./js'));
});

gulp.task('OrbitControls', function() {
    return gulp.src('./js/OrbitControls.js')
        .pipe(terser())
        .pipe(rename({extname: '.min.js'}))
        .pipe(gulp.dest('./js'));
});

gulp.task('pointCloud', function() {
    return gulp.src('./js/pointCloud.js')
        .pipe(terser())
        .pipe(rename({extname: '.min.js'}))
        .pipe(gulp.dest('./js'));
});

gulp.task('styleDemo', function() {
    return gulp.src('./js/styleDemo.js')
        .pipe(terser())
        .pipe(rename({extname: '.min.js'}))
        .pipe(gulp.dest('./js'));
});

gulp.task('styles', function () {
    return gulp.src('./scss/styles.scss')
        .pipe(wait(250))
        .pipe(sass({outputStyle: 'compressed'}).on('error', sass.logError))
        .pipe(gulp.dest('./css'));
});

gulp.task('watch', function() {
    gulp.watch('./js/scripts.js', gulp.series('scripts'));
    gulp.watch('./js/classifierDemo.js', gulp.series('classifierDemo'));
    gulp.watch('./js/depthDemo.js', gulp.series('depthDemo'));
    gulp.watch('./js/pointCloud.js', gulp.series('pointCloud'));
    gulp.watch('./js/OrbitControls.js', gulp.series('OrbitControls'));
    gulp.watch('./js/styleDemo.js', gulp.series('styleDemo'));
    gulp.watch('./scss/styles.scss', gulp.series('styles'));
});
