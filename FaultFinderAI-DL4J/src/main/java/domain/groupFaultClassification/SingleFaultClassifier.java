package domain.groupFaultClassification;

import java.io.IOException;
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
import faulttypes.SingleFaultFactory;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;

public class SingleFaultClassifier {

	private FaultClassifier classifier;
	private SingleFaultFactory factory;
	private int scoreIterations;
	private String fileName;
	private int nInputs;
	private int nLabels;
	/**
	 * since the labels are very asymmetric i.e. <br>
	 * wire position is 112*6 possible locations I <br>
	 * think the trainingSamples should be dependent <br>
	 * on the size of the faultLabel
	 */
	private int trainingSamples;
	private int faultType;
	private MultiLayerNetwork model;
	private FaultRecordScalerStrategy strategy;

	public SingleFaultClassifier(int faultType, int scoreIterations, FaultRecordScalerStrategy strategy) {
		this.faultType = faultType;
		this.scoreIterations = scoreIterations;
		this.strategy = strategy;
		init();

	}

	private void init() {
		initFault();
		initVars();

		// this.model = ModelFactory.simpleDNNII(this.nInputs, this.nLabels);
		// this.model = ModelFactory.simpleNN(this.nInputs, this.nLabels);
		// this.classifier = new FaultClassifier(model);
		loadModel();

	}

	private void initFault() {
		this.factory = new SingleFaultFactory();
		this.factory.getFault(faultType);
	}

	private void initVars() {
		this.fileName = factory.getFaultName();
		this.nInputs = factory.getFeatureArray().length;
		this.nLabels = factory.getReducedLabel().length;
		this.trainingSamples = this.nLabels * 1000;
	}

	public void loadModel() {
		String fileName = "models/SingleFaultClassifying" + this.fileName + ".zip";

		// if ((new File(fileName)).exists()) {
		// // initialize the classifier with the saved model
		// try {
		// this.classifier = new FaultClassifier(fileName);
		// } catch (IOException e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }
		// } else {
		// initialize the classifier with a fresh model
		this.model = ModelFactory.simpleOriginalCNN(this.nLabels);
		this.classifier = new FaultClassifier(model);
		// }

	}

	public void train() throws IOException {
		// set up a local web-UI to monitor the training available at
		// localhost:9000
		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		// additionally print the score on every iteration
		classifier.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(scoreIterations));
		uiServer.attach(statsStorage);
		classifier.train(this.nLabels, 1, trainingSamples, 1, new SingleClassifierRecordReader(this.faultType),
				strategy);

		// save the trained model
		classifier.save("models/SingleFaultClassifying" + fileName + ".zip");
	}

	public void evaluate() {
		// evaluate the classifier
		Evaluation evaluation = classifier.evaluate(this.nLabels, 1, 10000,
				new SingleClassifierRecordReader(this.faultType), strategy);
		System.out.println(evaluation.stats(false, true));
	}

	public void exit() {
		// press enter to exit the program
		// this will tear down the web ui
		Scanner sc = new Scanner(System.in);
		System.out.println("Press enter to exit.");
		sc.nextLine();

		System.exit(0);
	}

	public static void runLoop() throws IOException {
		for (int i = 0; i < 7; i++) {// 6 total faults, including fault #4
			// nofault
			if (i == 4) {// skip no fault do not need to classify something that
				// is not there
				continue;
			}
			SingleFaultClassifier sClassifier = new SingleFaultClassifier(i, 5000, new MinMaxStrategy());
			System.out.println("################## " + sClassifier.getFaultName() + " ##################");
			sClassifier.train();
			sClassifier.evaluate();
			// sClassifier.exit();

		}
	}

	public String getFaultName() {
		return this.fileName;
	}

	public static void main(String args[]) throws IOException {
		// faultType should never = 4, this is noFault
		// int faultType = 0;
		//
		// SingleFaultClassifier sClassifier = new
		// SingleFaultClassifier(faultType, 1000, new MinMaxStrategy());
		// System.out.println("################## " + sClassifier.getFaultName()
		// + " ##################");
		// sClassifier.train();
		// sClassifier.evaluate();
		// sClassifier.exit();
		runLoop();

	}
}
