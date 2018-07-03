package test;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.eval.Evaluation;

import client.FaultClassifier;
import client.ModelFactory;
import faultrecordreader.ReducedFaultRecordReader;

import java.util.Scanner;
import java.io.IOException;
import java.io.File;

public class FaultClassifierTest {
    public static void main (String args[]) throws IOException{
	// the model is stored here
	String fileName = "models/cnn_simple.zip";
	
	FaultClassifier classifier;
	// check if a saved model exists
	if ((new File(fileName)).exists()) {
	    // initialize the classifier with the saved model
	    classifier = new FaultClassifier(fileName);
	} else {
	    // initialize the classifier with a fresh model
	    MultiLayerNetwork model = ModelFactory.simpleCNN(14);

	    classifier = new FaultClassifier(model);
	}
	
        // set up a local web-UI to monitor the training available at localhost:9000
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        // additionally print the score on every iteration
        classifier.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));
        uiServer.attach(statsStorage);

	// train the classifier for a number of checkpoints and save the model after each checkpoint
	int checkPoints = 1;
	for (int i=0; i<checkPoints; i++) {
	    // train the classifier
	    classifier.train(20, 10000, 1, new ReducedFaultRecordReader());

	    // save the trained model
	    classifier.save(fileName);
	}

        // evaluate the classifier
        Evaluation evaluation = classifier.evaluate(1, 10000, new ReducedFaultRecordReader());
        System.out.println(evaluation.stats());

        // press enter to exit the program
        // this will tear down the web ui
        Scanner sc = new Scanner(System.in);
        System.out.println("Press enter to exit.");
        sc.nextLine();

        System.exit(0);
    }
}
