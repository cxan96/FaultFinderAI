package domain.groupFaultClassification;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Scanner;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

import client.FaultClassifier;
import client.ModelFactory;
import faultrecordreader.ReducedFaultRecordReader;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;

public class FaultClassifierTest {
	public static void main(String args[]) throws IOException {
		// the model is stored here
		int scoreIterations = 100;

		String fileName = "models/finallyNewSigmoid.zip";
		boolean reTrain = false;
		FaultClassifier classifier;
		// check if a saved model exists
		if ((new File(fileName)).exists()) {
			System.out.println("remodel");
			// initialize the classifier with the saved model
			classifier = new FaultClassifier(fileName);
		} else {
			// initialize the classifier with a fresh model
			MultiLayerNetwork model = ModelFactory.simpleCNN(14);

			classifier = new FaultClassifier(model);
		}
		FaultRecordScalerStrategy strategy = new MinMaxStrategy();

		// set up a local web-UI to monitor the training available at
		// localhost:9000
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		// additionally print the score on every iteration
		classifier.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(scoreIterations));
		uiServer.attach(statsStorage);

		// train the classifier for a number of checkpoints and save the model
		// after each checkpoint
		int checkPoints = 0;
		for (int i = 0; i < checkPoints; i++) {
			// train the classifier
			classifier.train(1, 10000, 1, new ReducedFaultRecordReader(), strategy);

			DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
			LocalDateTime now = LocalDateTime.now();

			// save the trained model
			classifier.save(fileName);
			// String altFileName = DomainUtils.getDropboxLocal() +
			// dtf.format(now) + "smallTests.zip";

			// classifier.save(altFileName);

			System.out.println("#############################################");
			System.out.println("Last checkpoint " + i + " at " + dtf.format(now));
			System.out.println("#############################################");

		}

		// evaluate the classifier
		Evaluation evaluation = classifier.evaluate(1, 10000, new ReducedFaultRecordReader(), strategy);
		System.out.println(evaluation.stats());
		// // lets compare recall here
		// int tPositive = 0;
		// int fNegative = 0;
		// for (int i = 0; i < 10; i++) {
		// FaultFactory factory = new FaultFactory();
		// factory.getFault(1);
		// System.out.println("Actual label: " +
		// Arrays.toString(factory.getReducedLabel()));

		// INDArray predictionsAtXYPoints =
		// classifier.output(factory.getFeatureVector());
		// int[] predictionArray = predictionsAtXYPoints.toIntVector();
		// System.out.println("Predicted label: " +
		// Arrays.toString(predictionArray));

		// int predictionIndex = 0;
		// int trueIndex = factory.getReducedFaultIndex();
		// for (int j = 0; j < predictionArray.length; j++) {
		// if (predictionArray[j] == 1) {
		// predictionIndex = j;
		// }
		// }
		// if ((predictionIndex - trueIndex) != 0) {
		// fNegative++;
		// } else {
		// tPositive++;
		// }
		// System.out.println("##############################");

		// }
		// System.out.println(tPositive + " " + fNegative);
		// System.out.println("Recall is = " + ((double) tPositive / ((double)
		// (tPositive + fNegative))));

		// press enter to exit the program
		// this will tear down the web ui
		Scanner sc = new Scanner(System.in);
		System.out.println("Press enter to exit.");
		sc.nextLine();

		System.exit(0);
	}
}
