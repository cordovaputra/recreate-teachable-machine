// const tf = require('@tensorflow/tfjs-node')

/* --------------------------------------------------------------------------------
A. REFERENCE & CREATE TEACHABLE MACHINE UI AND DATA COLLECTION FUNCTION
- Status: Indikator informasi training status
- Video : Visualisasi gambar yang akan di render untuk diproses datanya
*/
const STATUS = document.getElementById('status');               
const VIDEO = document.getElementById('webcam');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224; 
const MOBILE_NET_INPUT_HEIGHT = 224; 
const STOP_DATA_GATHERING_PROCESS = -1; //-1 means, stop the process
const CLASS_NAMES = [];

ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

//Reference & Create Data Collector Buttons in two classes from HTML to Javascript
let dataCollectorButtons = document.querySelectorAll('button.dataCollector');
for(let i = 0; i < dataCollectorButtons.length; i++) {
    dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
    dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);

    //Add human readable names for the classes
    CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}
/* --------------------------------------------------------------------------------*/

/* --------------------------------------------------------------------------------
B. DEFINE STATES VARIABLE
- gatherDataState = monitor state proses pengambilan data sedang aktif or not (button pressed or not)
- videPlaying = monitor state apakah video web cam berfungsi dan aktif or not
- trainingData = store data input dan output
- exampleCount = berapa sample data yang diinput di masing-masing class
- predict = monitor state prediksi sedang aktif or not
*/
let mobilenet = undefined; 
let gatherDataState = STOP_DATA_GATHERING_PROCESS;
let videoPlaying = false; 
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false; 
let mobileNetBase = undefined;

//Add Custom Console Printing Capabilities
function customConsolePrint(line) {
    let p = document.createElement('p');
    p.innerText = line; 
    document.body.appendChild(p);
}

/* --------------------------------------------------------------------------------*/

/* --------------------------------------------------------------------------------
C. INTEGRATE & LOAD DATA menggunakan Layer Models MobileNet
- tf.zeros  = Metode 'warmup' yang bertujuan untuk improve akurasi dari ML model
- mobilenet = variabel yang wrapping graph model dari web-server
*/
async function loadDataModel(){
    const URL = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json'
    mobilenet = await tf.loadLayersModel(URL);
    STATUS.innerText = 'Machine Learning model loaded successfully!';
    //mobilenet.summary(null, null, customConsolePrint); //print summary with custom console

    const layer = mobilenet.getLayer('global_average_pooling2d_1');
    mobileNetBase = tf.model(
        {
            inputs  : mobilenet.inputs, 
            outputs : layer.output
        }
    );
    mobileNetBase.summary();

    //Warming up te model by passing zeros through it, once
    tf.tidy(function(){
        let answer = mobileNetBase.predict
        (tf.zeros
            (
                [   1, 
                    MOBILE_NET_INPUT_HEIGHT, 
                    MOBILE_NET_INPUT_WIDTH, 
                    3
                ]
            )
        );
        console.log(answer.shape);
    });
}
loadDataModel();
/* --------------------------------------------------------------------------------*/

/* --------------------------------------------------------------------------------
D. CREATE MODEL ARCHITECTURE
- model with inputShape: Input Layer
- model with softmax: Output Layer
*/
let model = tf.sequential();
model.add(tf.layers.dense({inputShape: [1280], units: 64, activation: 'relu'}));
model.add(tf.layers.dense({units: CLASS_NAMES.length, activation: 'softmax'}));

model.summary();

model.compile({
    optimizer: 'adam',
    loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy' : 'categoricalCrossentropy',
    metrics: ['accuracy']
});
/* --------------------------------------------------------------------------------*/

/* --------------------------------------------------------------------------------
E. CONFIGURE AND ENABLE WEBCAM SERVICE
*/
function successfullyGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}
function enableCam() {
    if (successfullyGetUserMedia()) {
        //getUserMedia Parameters:
        const constraints = {
            video: true, 
            width: 640,
            height: 480
        };
        //Enable webcam's video stream
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            VIDEO.srcObject = stream; 
            VIDEO.addEventListener('loadeddata', function(){
                videoPlaying = true; 
                ENABLE_CAM_BUTTON.classList.add('removed');
            });
        });
        //If getUserMedia failed
    } else {
        console.warn('getUserMedia() is not supported by your browser');
    }
}
/* --------------------------------------------------------------------------------*/

/* --------------------------------------------------------------------------------
F. HANDLE DATA GATHER BUTTON STATE in EACH CLASSES
*/
function gatherDataForClass(){
    let classNumber = parseInt(this.getAttribute('data-1hot'));
    gatherDataState = (gatherDataState === STOP_DATA_GATHERING_PROCESS) ? classNumber : STOP_DATA_GATHERING_PROCESS;
    dataGatherLoop();
}

function calculateFeaturesOnCurrentFrame() {
    return tf.tidy(function() {

        //Grab a frame from webcam and store as Tensors
        let videoFrameAsTensor = tf.browser.fromPixels(VIDEO); 
                
        let resizedTensorFrame = tf.image.resizeBilinear
        (videoFrameAsTensor, 
            [
                MOBILE_NET_INPUT_HEIGHT,
                MOBILE_NET_INPUT_WIDTH
            ], 
            true
        );

        let normalizedTensorFrame = resizedTensorFrame.div(255);
        return mobileNetBase.predict(normalizedTensorFrame.expandDims()).squeeze();
    });
}

function dataGatherLoop() {
    //Data gathering process runs if webcam is active and data gathering button is pressed
    if (videoPlaying && gatherDataState !== STOP_DATA_GATHERING_PROCESS) {
        
        //Ensure tensors are cleaned up
        let imageFeatures = calculateFeaturesOnCurrentFrame();

        trainingDataInputs.push(imageFeatures);
        trainingDataOutputs.push(gatherDataState);

        // Initialize array index element if currently undefined
        if (examplesCount[gatherDataState] === undefined) {
            examplesCount[gatherDataState] = 0;
        }
        examplesCount[gatherDataState]++;

        STATUS.innerText = '';
        for (let n = 0; n < CLASS_NAMES.length; n++) {
            STATUS.innerText += CLASS_NAMES[n] + '  data count: ' + examplesCount[n] + '. ';
        }
        window.requestAnimationFrame(dataGatherLoop);
    }
}
/* --------------------------------------------------------------------------------*/

/* --------------------------------------------------------------------------------
G. HANDLE MODEL TRAINING AND PREDICTION
*/
async function trainAndPredict(){
    predict = false; 
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
    let oneHotOutputs   = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
    let inputsAsTensor  = tf.stack(trainingDataInputs);

    let results = await model.fit
    (
        inputsAsTensor, 
        oneHotOutputs, 
        {
            shuffle     : true,
            batchSize   : 5, 
            epochs      : 5,
            callbacks   : {
                onEpochEnd: logProgress
            }
        }
    );
    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();
    predict = true; 
    
    // Create Combined Model for Download
    let combinedModel = tf.sequential();
    combinedModel.add(mobileNetBase);
    combinedModel.add(model);

    combinedModel.compile({
        optimizer   : 'adam',
        loss        : (CLASS_NAMES.length === 2) ? 'binaryCrossentropy' : 'categoricalCrossentropy'
    });

    combinedModel.summary();
    await combinedModel.save('downloads://my-model');
    predictLoop();
}

function logProgress(epoch, logs) {
    console.log('Data untuk epoch: ' + epoch, logs);
}

function predictLoop(){
    if (predict) {
        tf.tidy(function() {

            let imageFeatures   = calculateFeaturesOnCurrentFrame();
            let prediction      = model.predict(imageFeatures.expandDims()).squeeze();
            let highestIndex    = prediction.argMax().arraySync();
            let predictionArray = prediction.arraySync();

            STATUS.innerText = 'Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + 
            Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
        });
        
        window.requestAnimationFrame(predictLoop);
    }
}
/* --------------------------------------------------------------------------------*/

/* --------------------------------------------------------------------------------
H. CONFIGURE RESET FUNCTIONALITY
*/
function reset() {
    predict = false; 
    examplesCount.splice(0);

    for (let i = 0; i < trainingDataInputs.length; i++) {
        trainingDataInputs[i].dispose();
    }
    trainingDataInputs.splice(0);
    trainingDataOutputs.splice(0);
    STATUS.innerText = 'Reset successfull. No data collected';

    console.log('Tensors in memory: ' + tf.memory().numTensors);
}