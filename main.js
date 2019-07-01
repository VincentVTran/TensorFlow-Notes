const tf = require('@tensorflow/tfjs');

function basics() {
    //Tensor refers to type of matrices dimensions (Ex: Scalar 2, Vector [2,3,4], Matrix [1 2] [2,4])
    tf.tensor([1,2,3,4]);

    console.log("4x1 data set: ")
    tf.tensor([0,0,127,255], [4,1]).print(); //[Data values] , [Rows,Column]

    console.log("2x2 data set: ")
    tf.tensor([0,0,127,255], [2,2],'int32').print();   //[Data values] , [Rows,Column] , 'Data Type'

    console.log("2x2x2 (2 2x2) data set: ")
    tf.tensor([0,0,127,255,122,133,0,122], [2,2,2]).print();  
}


function setUp() { //Using variables to create tensor
    const dataSet = [];
    for(let i = 0;i<15;i++){
        dataSet[i] = Math.random()*100;
    }
    dataDimensions = [5,3];

    let tens = tf.tensor(dataSet,dataDimensions); //Creates the tensor

    tens.data().then(extracted => {console.log(extracted)}); //Turns data into a promise; Once promise is recieved, it will enter 'then' function

    const varTens = tf.variable(tens); //Turns it into a variable tensor (Can now be changed [mutiable])
}

function tensor3dAdding(){
    const dataSet = [];
    const dimensions = [2,5,3];

    for(let i = 0;i<30;i++){
        dataSet[i] = Math.random()*100;
    }

    const tens1 = tf.tensor3d(dataSet,dimensions,'int32'); //Creates tensor #1
    tens1.print();
    const tens2 = tf.tensor3d(dataSet,dimensions,'int32'); //Creates tensor #2
    tens2.print();

    const tensSum = tf.add(tens1,tens2); //Adds the tensor together
    tensSum.print();
}

function tensorMultiplication(){
    const tens1 = tf.tensor([1,2,3,4,5,6],[2,3],'int32'); //Creates tensor #1
    tens1.print();
    const tens2 = tf.tensor([1,2,3,4,5,6],[3,2],'int32'); //Creates tensor #2
    tens2.print();

    const tensMultiply = tf.matMul(tens1,tens2); //(Rule when multiplying Tensors) [Columns in tens1 must match row in tens2]
    tensMultiply.print();  

    tens1.transpose()//Flips the dimension to opposite
}


//tf.tidy(tensorMemory()); //Disposes all tensor after 'tensorMemory' function ends
function tensorMemory(){
    const tens1 = tf.tensor([1,2,3,4,5,6],[2,3],'int32');
    const tens2 = tf.tensor([1,2,3,4,5,6],[3,2],'int32'); 

    tens1.dispose();
    tens2.dispose();
    console.log(tf.memory().numTensors); //Output: 0 tensors in memory
}
//basics();
//setUp();
//tensor3dAdding();
//tensorMultiplication();
//tensorMemory();

function createNeuralNetwork(){
    const neuralNetwork = tf.sequential(); //Creates overall neural network architecture

    //Layers
    const configHiddenLayer = { //Configuration for layers
        units: 4, //# of nodes
        inputShape: [2], //# of Inputs required
        activation: 'sigmoid', //Type of activation function (Controls number range)
    }

    const configOutputLayer = { //Configuration for layers
        units: 4, //# of nodes
        activation: 'sigmoid', //Type of activation function (Controls number range)
    }
    const hidden = tf.layers.dense(configHiddenLayer); //Creating a layer that connects all nodes to each other
    const output = tf.layers.dense(configOutputLayer);

    neuralNetwork.add(hidden); //Adding hidden layer to network
    neuralNetwork.add(output);

    //Optimizer
    const sgdOpt = tf.train.sgd(0.1);
    const configOptimizer = {
        optimizer: sgdOpt,
        loss: tf.losses.meanSquaredError,
    }
    neuralNetwork.compile(configOptimizer);
}

createNeuralNetwork();