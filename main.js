const ts = require('@tensorflow/tfjs');

//Tensor refers to type of matrices dimensions (Ex: Scalar 2, Vector [2,3,4], Matrix [1 2] [2,4])
ts.tensor([1,2,3,4]);

console.log("4x1 data set: ")
ts.tensor([0,0,127,255], [4,1]).print(); //[Data values] , [Rows,Column]

console.log("2x2 data set: ")
ts.tensor([0,0,127,255], [2,2],'int32').print();   //[Data values] , [Rows,Column] , 'Data Type'

console.log("2x2x2 (2 2x2) data set: ")
ts.tensor([0,0,127,255,122,133,0,122], [2,2,2]).print();  

function setUp() { //Using variables to create tensor
    const dataSet = [];
    for(let i = 0;i<15;i++){
        dataSet[i] = Math.random()*100;
    }
    dataDimensions = [5,3];
    ts.tensor(dataSet,dataDimensions).print();
}

setUp();