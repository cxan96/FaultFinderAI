package domain.groupFaultClassification;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.datavec.api.records.reader.RecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

import client.FaultClassifier;
import client.ModelFactory;
import faultrecordreader.KunkelPetersFaultRecorder;
import faults.FaultNames;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;

public class FaultClassifierTestLooped {
	public static void main(String args[]) throws IOException {

		List<FaultNames> aList = new ArrayList<>();
		aList.add(FaultNames.CHANNEL_ONE);
		aList.add(FaultNames.CHANNEL_TWO);
		aList.add(FaultNames.CHANNEL_THREE);
		aList.add(FaultNames.CONNECTOR_E);
		aList.add(FaultNames.CONNECTOR_THREE);
		aList.add(FaultNames.CONNECTOR_TREE);
		aList.add(FaultNames.FUSE_A);
		aList.add(FaultNames.FUSE_B);
		aList.add(FaultNames.FUSE_C);
		for (int moreSaves = 1; moreSaves < 4; moreSaves++) {

			for (int SL = 1; SL < 7; SL++) {
				for (FaultNames faultNames : aList) {
					int scoreIterations = 10000;

					String fileName = "models/binary_classifiers/SL" + SL + "/" + faultNames + "_save" + moreSaves
							+ ".zip";
					System.out.println(fileName);
					FaultClassifier classifier;
					// check if a saved model exists
					if ((new File(fileName)).exists()) {
						System.out.println("remodel");
						// initialize the classifier with the saved model
						classifier = new FaultClassifier(fileName);
					} else {
						// initialize the classifier with a fresh model
						MultiLayerNetwork model = ModelFactory.deeperCNN(2);

						classifier = new FaultClassifier(model);
					}
					FaultRecordScalerStrategy strategy = new MinMaxStrategy();

					// set up a local web-UI to monitor the training available
					// at
					// localhost:9000
					UIServer uiServer = UIServer.getInstance();
					StatsStorage statsStorage = new InMemoryStatsStorage();
					// additionally print the score on every iteration
					classifier.setListeners(new StatsListener(statsStorage),
							new ScoreIterationListener(scoreIterations));
					uiServer.attach(statsStorage);

					// train the classifier for a number of checkpoints and save
					// the
					// model
					// after each checkpoint
					RecordReader recordReader = new KunkelPetersFaultRecorder(SL, 10, faultNames, false);
					int checkPoints = 5;
					for (int i = 0; i < checkPoints; i++) {
						// train the classifier
						classifier.train(2, 1, 5000, 1, recordReader, strategy);

						DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
						LocalDateTime now = LocalDateTime.now();

						// save the trained model
						classifier.save(fileName);

						System.out.println("#############################################");
						System.out.println("Last checkpoint " + i + " at " + dtf.format(now));
						System.out.println("#############################################");

					}
					System.out.println("Evaluation for " + faultNames);
					// evaluate the classifier
					Evaluation evaluation = classifier.evaluate(2, 1, 10000, recordReader, strategy);
					System.out.println(evaluation.stats());

				}
			}
		}
		// press enter to exit the program
		// this will tear down the web ui
		Scanner sc = new Scanner(System.in);
		System.out.println("Press enter to exit.");
		sc.nextLine();

		System.exit(0);
	}
}
