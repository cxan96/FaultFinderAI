package test;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import client.FaultClassifier;
import client.ModelFactory;
import faultrecordreader.ReducedFaultRecordReader;

public class FaultClassifierTest {
    public static void main (String args[]) {
	// get the desired model
	MultiLayerNetwork model = ModelFactory.simpleCNN(13);

	// initialize the fault classifier
	FaultClassifier classifier = new FaultClassifier(model);

	// add a listener to monitor training
	classifier.setListeners(new ScoreIterationListener(1));

	// train the classifier
	classifier.train(20, 5000, new ReducedFaultRecordReader());

	// evaluate the classifier
	String stats = classifier.evaluate(1, 5000, new ReducedFaultRecordReader());
	System.out.println(stats);

	System.exit(0);
    }
}
