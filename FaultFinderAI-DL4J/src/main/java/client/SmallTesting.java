package client;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

import faultrecordreader.ReducedFaultRecordReader;
import strategies.FaultRecordScalerStrategy;
import strategies.StandardizeMinMax;
import utils.DomainUtils;

public class SmallTesting {
	public static void main(String args[]) throws IOException {
		int scoreIterations = 1000;

		FaultClassifier classifier;
		// initialize the classifier with a fresh model
		MultiLayerNetwork model = ModelFactory.simpleCNN(14);
		FaultRecordScalerStrategy strategy = new StandardizeMinMax(0.05);

		classifier = new FaultClassifier(model);
		// set up a local web-UI to monitor the training available at
		// localhost:9000
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		// additionally print the score on every iteration
		classifier.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(scoreIterations));
		uiServer.attach(statsStorage);

		// train the classifier for a number of checkpoints and save the model
		// after each checkpoint
		int checkPoints = 1;
		for (int i = 0; i < checkPoints; i++) {
			// train the classifier
			classifier.train(20, 10000, 5, new ReducedFaultRecordReader(), strategy);
			// classifier.train(5, 50, 10, new ReducedFaultRecordReader());

			DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
			LocalDateTime now = LocalDateTime.now();

			// save the trained model
			// classifier.save(fileName);
			String altFileName = DomainUtils.getDropboxLocal() + dtf.format(now) + "OldNormalization.zip";

			classifier.save(altFileName);

			System.out.println("#############################################");
			System.out.println("Last checkpoint " + i + " at " + dtf.format(now));
			System.out.println("#############################################");

		}

		// evaluate the classifier
		Evaluation evaluation = classifier.evaluate(1, 10000, new ReducedFaultRecordReader(), strategy);
		System.out.println(evaluation.stats());
		System.exit(0);
	}
}
