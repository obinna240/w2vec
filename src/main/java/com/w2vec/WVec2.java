package com.w2vec;

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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collection;

/**
 * Created by agibsonccc on 10/9/14.
 *
 * Neural net that processes text into wordvectors. See below url for an in-depth explanation.
 * https://deeplearning4j.org/word2vec.html
 */
public class WVec2 {

    private static Logger log = LoggerFactory.getLogger(WVec2.class);

    public static void main(String[] args) throws Exception 
    {

        // Gets Path to Text file
        String filePath = "new.txt"; //new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();

        log.info("Load & Vectorize Sentences....");
        
        ////Get each sentence in the document
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(filePath);
        
        ////word2vec needs to be fed words and not documents per se,
        ////so we tokenize each sentence
        //System.out.println(iter.toString());
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
        Collection<String> lst = vec.wordsNearest("darcy", 10);
        System.out.println("10 Words closest to 'darcy': " + lst);

        System.out.println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
        System.out.println(vec.getWordVector("darcy").length);
        
        INDArray indArray = vec.getWordVectorMatrix("darcy");
        // TODO resolve missing UiServer
//        UiServer server = UiServer.getInstance();
//        System.out.println("Started on port " + server.getPort());
       System.out.println( indArray);
       System.out.println( "##################################################################");
       WeightLookupTable<VocabWord> x = vec.getLookupTable();
       System.out.println(x);
      // FileUtils.writeLines(new File("f.txt"), x.toString(), null, true);
    }
}