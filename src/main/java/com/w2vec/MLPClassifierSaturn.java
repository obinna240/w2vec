package com.w2vec;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

public class MLPClassifierSaturn 
{
	public static void main(String[] args) throws Exception
	{
		Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
		//batchsize controls how many examples are gathered from disk and passed
		//on to the model for training in a batch. As the model is training we should
		//see output on the console 
		int batchSize = 40;
		int seed = 123;
		double learningRate = 0.005;
		//Number of epochs -- full passes over the entire dataset. 
		//Many times we train over multiple epochs until convergence is reached
		int nEpochs = 30;
		int numInputs = 2;
		int numOutputs = 2;
		int numHiddenNodes = 20;
		
		//final String fileNameTrain = new ClassPathResource("/saturn_data_train.csv").getFile().getPath();
		//final String fileNameTest = new ClassPathResource("/saturn_data_eval.csv").getFile().getPath();
		
		final String fileNameTrain = "saturn_data_train.csv";
		final String fileNameTest = "saturn_data_eval.csv";
		
		//load the training data
		RecordReader rr = new CSVRecordReader();
		FileSplit fsplit = new FileSplit(new File(fileNameTrain));
		rr.initialize(fsplit);
		
		
		
		//This works with the recordReaders to take the produced NDArrays 
		//for each record and create mini-batches of NDArrays for training.
		//(RecordReader recordReader, int batchSize, int labelIndex, int numPossibleLabels)
		//We parse the data with a recordReader and we tell the iterator the record reader, the batchSize
		//which is the number of batches, the label inidex which is column 0 and the number of possible labels
		//which in this case is 2, (0 and 1)
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0,2);
		
		
		//load the test/evaluation data
		RecordReader rrTest = new CSVRecordReader();
		FileSplit fsplit2 = new FileSplit(new File(fileNameTest));
		rrTest.initialize(fsplit2);
		
		//(RecordReader recordReader, int batchSize, int labelIndex, int numPossibleLabels)
		//We parse the data with a recordReader and we tell the iterator the record reader, the batchSize
		//which is the number of batches, the label inidex which is column 0 and the number of possible labels
		//which in this case is 2, (0 and 1)
		DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0,2);
		
		//build the model
		//list() is a listBuilder for building the layers	
		/**
		 * The First Hidden Layer - Takes on raw values from the input that we have produced. 
		 * These are our input values (normalized). The number of neurons needs to be the same as the number
		 *  of independent variables or columns in our input vector. I.e .nIn(numInputs)
		 */
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).iterations(1)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(learningRate).
	updater(Updater.NESTEROVS).momentum(0.9).list().
	layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(/*number of hidden nodes in next layer*/numHiddenNodes).weightInit(/* we also initaialize the weights*/WeightInit.XAVIER)
			.activation(Activation.RELU).build()).
	layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).weightInit(WeightInit.XAVIER)
			.activation(Activation.SOFTMAX).nIn(numHiddenNodes).nOut(numOutputs).build())
			.pretrain(false).backprop(true).build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(10));
		for(int n=0;n<nEpochs;n++)
		{
			model.fit(trainIter);
		}
		System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);

            eval.eval(lables, predicted);

        }


        System.out.println(eval.stats());
        //------------------------------------------------------------------------------------
        //Training is complete. Code that follows is for plotting the data & predictions only


        double xMin = -15;
        double xMax = 15;
        double yMin = -15;
        double yMax = 15;

        //Let's evaluate the predictions at every point in the x/y input space, and plot this in the background
        int nPointsPerAxis = 100;
        double[][] evalPoints = new double[nPointsPerAxis*nPointsPerAxis][2];
        int count = 0;
        for( int i=0; i<nPointsPerAxis; i++ ){
            for( int j=0; j<nPointsPerAxis; j++ ){
                double x = i * (xMax-xMin)/(nPointsPerAxis-1) + xMin;
                double y = j * (yMax-yMin)/(nPointsPerAxis-1) + yMin;

                evalPoints[count][0] = x;
                evalPoints[count][1] = y;

                count++;
            }
        }

        INDArray allXYPoints = Nd4j.create(evalPoints);
        INDArray predictionsAtXYPoints = model.output(allXYPoints);

        //Get all of the training data in a single array, and plot it:
        rr.initialize(new FileSplit(new File(fileNameTrain)));
        rr.reset();
        int nTrainPoints = 500;
        trainIter = new RecordReaderDataSetIterator(rr,nTrainPoints,0,2);
        DataSet ds = trainIter.next();
        PlotUtil.plotTrainingData(ds.getFeatures(), ds.getLabels(), allXYPoints, predictionsAtXYPoints, nPointsPerAxis);


        //Get test data, run the test data through the network to generate predictions, and plot those predictions:
        rrTest.initialize(new FileSplit(new File(fileNameTest)));
        rrTest.reset();
        int nTestPoints = 100;
        testIter = new RecordReaderDataSetIterator(rrTest,nTestPoints,0,2);
        ds = testIter.next();
        INDArray testPredicted = model.output(ds.getFeatures());
        PlotUtil.plotTestData(ds.getFeatures(), ds.getLabels(), testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis);

        System.out.println("****************Example finished********************");
    }
}