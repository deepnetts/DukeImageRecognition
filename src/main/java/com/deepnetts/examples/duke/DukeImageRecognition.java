package com.deepnetts.examples.duke;

import deepnetts.util.LabelProbabilities;
import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.data.TrainTestSplit;
import deepnetts.eval.ClassificationMetrics;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.eval.ConfusionMatrix;
import deepnetts.net.layers.Filters;
import deepnetts.net.loss.LossType;
import deepnetts.util.ImagePreprocessing;
import deepnetts.util.ImageResize;
import deepnetts.util.RandomGenerator;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.eval.EvaluationMetrics;
import javax.visrec.ri.ml.classification.ImageClassifierNetwork;

/**
 * Duke Java mascot image recognition.
 * This examples shows how to use convolutional neural network for binary classification of images.
 *
 * Data set contains 114 images of Duke and Non-Duke examples.
 * 
 * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/getting-started
 * 
 * @see ConvolutionalNetwork
 * @see ImageSet
 * 
 */
public class DukeImageRecognition {

    static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());  
    
    public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {

        // width and height of input images
        int imageWidth = 64;
        int imageHeight = 64;
        
        // paths to training images
        // ovi fajlovi treba sami da se kreiraju! nadji resenje, posebno za zero mean
        // mozda samo setZeroMean boolean a ne poziv metode, mozda i builder za ImageSet
        String dataSetPath = "DukeSet"; // path to folder with images
        String labelsFile = dataSetPath + "/labels.txt"; // path to plain file with list of labels
        String trainingFile = dataSetPath + "/index.txt"; // path to plain txt file with list of images and coresponding labels


        RandomGenerator.getDefault().initSeed(123); // fix global random generator to get repeatable training results
    //    DeepNetts.getInstance();
    
        // initialize image data set and preprocessing
        LOGGER.info("Loading images...");
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight); // napraviti builder? da zero mean radi i ucitavanje slika
        imageSet.setResizeStrategy(ImageResize.STRATCH);
        imageSet.setInvertImages(true);
        imageSet.loadLabels(new File(labelsFile));// automatski generisi na osnovu foldera
        imageSet.loadImages(new File(trainingFile)); // ucitaj na kraju 
        imageSet.zeroMean(); // ovo moze tek kad ucita sve slike
                             // da moze da ga kesira u .deepnatts dir preprocesiranog treba mi builder za to 
        
        LOGGER.info("Splitting data into training and test sets...");  
        TrainTestSplit trainTest = imageSet.trainTestSplit(0.7);

        // create a convolutional neural network arhitecture for binary image classification
        LOGGER.info("Creating a neural network...");
        ConvolutionalNetwork convNet = ConvolutionalNetwork.builder()
                .addInputLayer(imageWidth, imageHeight, 3)
                .addConvolutionalLayer(6, Filters.ofSize(3))
                .addMaxPoolingLayer(Filters.ofSize(2).stride(2))
                .addFullyConnectedLayer(16)
                .addOutputLayer(1, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .hiddenActivationFunction(ActivationType.LEAKY_RELU)
                .build();

        // set training options and run training
        LOGGER.info("Training the neural network using training set ...");        
        BackpropagationTrainer trainer = convNet.getTrainer(); // Get a trainer of the created convolutional network
        trainer.setStopError(0.03f)         // training should stop once the training error is below this value
               .setLearningRate(0.01f);     // amount of error to use for adjusting internal parameters in each training step
        trainer.train(trainTest.getTrainSet()); // run training

        LOGGER.info("Testing the trained neural network - computing various evaluation metrics using test set...");
        EvaluationMetrics testResults = convNet.test(trainTest.getTestSet());
        System.out.println(testResults);
            
        // get confusion matrix which contains details about correct and wrong classifications        
        ConfusionMatrix confusionMatrix = ((ClassificationMetrics)testResults).getConfusionMatrix(); 
        System.out.println(confusionMatrix);
        
        // save the trained neural network into a file
        LOGGER.info("Saving the trained neural network.");
        convNet.save("DukeImageClassifier.dnet");

        // how to use image recognizer for a new external image
        LOGGER.info("Recognizing an example duke image.");

        // enable image preprocessing for new images during inference - use same preprocessing that was used for training
        ((ImagePreprocessing)convNet.getPreprocessing()).setEnabled(true);
                     
        BufferedImage image = ImageIO.read(new File("DukeSet/duke/duke7.jpg"));
        ImageClassifier<BufferedImage> imageClassifier = new ImageClassifierNetwork(convNet);
        Map<String, Float> results = imageClassifier.classify(image); // result is a map with image labels as keys and coresponding probability
        LabelProbabilities labelProb = new LabelProbabilities(results);
        LOGGER.info(labelProb.toString());
     
        // shutdown the thread pool
        DeepNetts.shutdown();
    }

}