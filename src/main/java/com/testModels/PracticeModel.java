package com.testModels;

import java.io.BufferedOutputStream;
import java.io.FileNotFoundException;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
//import org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
//import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class PracticeModel
{
	public static void main(String[] args) throws FileNotFoundException
	{
		//write a model to disk or hdfs using ModelSerializer
		//BufferedOutputStream boss = new BufferedOutputStream(IOUtils);
		//read a model using modelserializer
		
		String filePath = "raw_sentences.txt"; //new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();

   
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(filePath);
        System.out.println(iter.toString());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();

        /*
            CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
            So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
            Additionally it forces lower case for all tokens.
         */
        t.setTokenPreProcessor(new CommonPreprocessor());

       
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

       
        vec.fit();
   
        //we use RecoredReaders and DatasetIterators to call NDArray for records and create mini-batches of NDArrays for
        //training
        //Build a layer oriented architecture
        //we build many single layers which when combined constitute a deep neural net
        //observe that we add each layer and configure it specifically
        MultiLayerConfiguration conf1 = new NeuralNetConfiguration.Builder().seed(1).iterations(1).
        		optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(1.0) 
        		.updater(Updater.NESTEROVS).momentum(0.9).list() //updaters are hyperparameters
        		.layer(0, new DenseLayer.Builder()
        			.nIn(120).nOut(10).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build()).
        		layer(1, 
        				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).weightInit(WeightInit.XAVIER).
        				activation(Activation.SOFTMAX).nIn(10).nOut(2).build()).pretrain(false).backprop(true).build();
     
	}
}
