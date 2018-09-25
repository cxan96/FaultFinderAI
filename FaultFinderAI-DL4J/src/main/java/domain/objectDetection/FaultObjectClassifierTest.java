package domain.objectDetection;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import org.datavec.api.records.reader.RecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

import client.ModelFactory;
import faultrecordreader.FaultObjectDetectionRecordReader;
import faults.FaultNames;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;

public class FaultObjectClassifierTest {
	public static void main(String args[]) throws IOException {
		// the model is stored here
		int scoreIterations = 1000;

		String fileName = "models/binary_classifiers/ComputationalGraphModel/TestII.zip";
		boolean reTrain = false;
		FaultObjectClassifier classifier;
		// check if a saved model exists
		if ((new File(fileName)).exists()) {
			System.out.println("remodel");
			// initialize the classifier with the saved model
			classifier = new FaultObjectClassifier(fileName);
		} else {
			// initialize the classifier with a fresh model
			// ComputationGraph model =
			// ModelFactory.deeperCNN(2).toComputationGraph();
			// ComputationGraph model = ModelFactory.computationGraphModelII(2);
			ComputationGraph model = ModelFactory.computationGraphModelYolo(2);

			classifier = new FaultObjectClassifier(model);
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

		// RecordReader recordReader = new KunkelPetersFaultRecorder(2, 10,
		// FaultNames.PIN_SMALL, true, true);
		RecordReader recordReader = new FaultObjectDetectionRecordReader(3, 10, FaultNames.CHANNEL_ONE, true, true, 112,
				6, 1, 56, 3);
		// (2, 10, FaultNames.PIN_SMALL, true, true);

		int checkPoints = 1;
		for (int i = 0; i < checkPoints; i++) {
			// train the classifier
			classifier.train(2, 1, 10000, 1, recordReader, strategy);

			DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
			LocalDateTime now = LocalDateTime.now();

			// save the trained model
			classifier.save(fileName);

			System.out.println("#############################################");
			System.out.println("Last checkpoint " + i + " at " + dtf.format(now));
			System.out.println("#############################################");

		}

		// evaluate the classifier
		Evaluation evaluation = classifier.evaluate(2, 1, 10000, recordReader, strategy);
		System.out.println(evaluation.stats());
	}
}
