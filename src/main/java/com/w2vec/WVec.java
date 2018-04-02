package com.w2vec;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.io.FileUtils;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.List;

/**
 * Created by agibsonccc on 10/9/14.
 *
 * Neural net that processes text into wordvectors. See below url for an in-depth explanation.
 * https://deeplearning4j.org/word2vec.html
 */
public class WVec {

    private static Logger log = LoggerFactory.getLogger(WVec.class);

    public static void main(String[] args) throws Exception 
    {

        // Gets Path to Text file
   String filePath = "health_wales.txt"; //new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();
       
       // String largeS = FileUtils.readFileToString(new File(filePath));
       // largeS = largeS.toLowerCase();
       // FileUtils.deleteQuietly(new File("health_wales.txt"));
        //FileUtils.write(new File("health_wales.txt"), largeS, "utf-8", true);
      //  log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(filePath);
      
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();

        /*
            CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
            So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
            Additionally it forces lower case for all tokens.
         */
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing word vectors to text file....");

        // Prints out the closest 10 words to "day". An example on what to do with these Word Vectors.
        log.info("Closest Words:");
        Collection<String> lst = vec.wordsNearest("doctor", 10);
        System.out.println("10 Words closest to 'health': " + lst);
       // System.out.println(vec.getElementsScore());
       // WeightLookupTable<VocabWord> vcd =  vec.getLookupTable();
        //System.out.println("vec matrix " +vec.getWordVectorMatrix("day"));
        //System.out.println("word vectors" +vec.getWordVector("day"));
        
        List<String> ls = vec.similarWordsInVocabTo("doctor", 0.75);
        System.out.println(ls);
       
        for(String s:lst)
        {
        	System.out.println("similarity between "+ s+" and health"+vec.similarity("health", s));
        }
        for(String s:ls)
        {
        	System.out.println("2. similarity between "+ s+" and health"+vec.similarity("health", s));
        }
       
        // TODO resolve missing UiServer
//        UiServer server = UiServer.getInstance();
//        System.out.println("Started on port " + server.getPort());
        
    }
}