package domain.groupFaultClassification;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

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

	private List<FaultNames> fautList;
	private int superLayer;

	private int savePoint;
	private int nFaults;

	private int checkPoints;
	private int scoreIterations;
	private FaultClassifier classifier;
	private FaultRecordScalerStrategy strategy;
	private RecordReader recordReader;
	private UIServer uiServer;
	private StatsStorage statsStorage;
	private String fileName;
	private String saveName;
	private int batchSize;

	public FaultClassifierTestLooped(int superLayer, int savePoint, int scoreIterations, int nFaults, int checkPoints,
			int batchSize) {
		this.superLayer = superLayer;
		this.savePoint = savePoint;
		this.scoreIterations = scoreIterations;
		this.nFaults = nFaults;
		this.checkPoints = checkPoints;
		this.batchSize = batchSize;
		makeList();
	}

	private void makeList() {
		fautList = new ArrayList<>();
		// fautList.add(FaultNames.CHANNEL_ONE);
		// fautList.add(FaultNames.CHANNEL_TWO);
		// fautList.add(FaultNames.CHANNEL_THREE);
		fautList.add(FaultNames.CONNECTOR_E);
		fautList.add(FaultNames.CONNECTOR_THREE);
		fautList.add(FaultNames.CONNECTOR_TREE);
		fautList.add(FaultNames.FUSE_A);
		fautList.add(FaultNames.FUSE_B);
		fautList.add(FaultNames.FUSE_C);

		fautList.add(FaultNames.DEADWIRE);
		fautList.add(FaultNames.HOTWIRE);
		fautList.add(FaultNames.PIN_BIG);
		fautList.add(FaultNames.PIN_SMALL);

	}

	public void runClassifier() throws IOException {
		for (FaultNames faultNames : fautList) {

			this.fileName = "models/binary_classifiers/IntegratedModel/" + faultNames.getSaveName() + "_save"
					+ (this.savePoint - 1) + ".zip";
			this.saveName = "models/binary_classifiers/IntegratedModel/" + faultNames.getSaveName() + "_save"
					+ this.savePoint + ".zip";
			System.out.println(fileName);

			// check if a saved model exists
			if ((new File(fileName)).exists()) {
				System.out.println("remodel");
				// initialize the classifier with the saved model
				this.classifier = new FaultClassifier(fileName);
			} else {
				// initialize the classifier with a fresh model
				MultiLayerNetwork model = ModelFactory.deeperCNN(2);

				this.classifier = new FaultClassifier(model);
			}
			this.strategy = new MinMaxStrategy();

			// set up a local web-UI to monitor the training available
			// at
			// localhost:9000
			this.uiServer = UIServer.getInstance();
			this.statsStorage = new InMemoryStatsStorage();
			// additionally print the score on every iteration
			classifier.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(scoreIterations));
			uiServer.attach(statsStorage);

			// train the classifier for a number of checkpoints and save
			// the
			// model
			// after each checkpoint
			this.recordReader = new KunkelPetersFaultRecorder(this.superLayer, this.nFaults, faultNames, false);
			for (int i = 0; i < this.checkPoints; i++) {
				// train the classifier
				classifier.train(2, 1, this.batchSize, 1, recordReader, strategy);

				DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
				LocalDateTime now = LocalDateTime.now();

				// save the trained model
				classifier.save(this.saveName);

				System.out.println("#############################################");
				System.out.println("Last checkpoint " + i + " at " + dtf.format(now));
				System.out.println("#############################################");

			}
			// this.uiServer.stop();
			// runEvaluation();
		}
	}

	public void runEvaluation() {
		// System.out.println("Evaluation for " + faultNames);
		// evaluate the classifier
		Evaluation evaluation = classifier.evaluate(2, 1, 10000, recordReader, strategy);
		System.out.println(evaluation.stats());
	}

	public static void main(String[] args) throws IOException {
		for (int moreSaves = 1; moreSaves < 2; moreSaves++) {
			// for (int SL = 1; SL < 7; SL++) {
			FaultClassifierTestLooped looped = new FaultClassifierTestLooped(1, moreSaves, 5000, 10, 10, 10000);
			looped.runClassifier();
			// }
		}
	}
}
