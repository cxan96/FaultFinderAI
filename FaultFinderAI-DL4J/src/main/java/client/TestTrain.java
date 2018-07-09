package client;

import java.io.IOException;

import org.deeplearning4j.eval.Evaluation;

import faultrecordreader.ReducedFaultRecordReader;
import utils.DomainUtils;

public class TestTrain {

	public static void test(String fileName) throws IOException {
		FaultClassifier classifier = new FaultClassifier(fileName);

		// classifier.train(1, 1, 1, new ReducedFaultRecordReader());

		// System.out.println("size = " +
		// classifier.geTrainingListeners().size());
		// for (TrainingListener tr : classifier.geTrainingListeners()) {
		// System.out.println(tr.getClass().getName());
		// }

		Evaluation evaluation = classifier.evaluate(1, 10000, new ReducedFaultRecordReader());
		System.out.println(evaluation.stats());
		System.out.println("\n \n  " + evaluation.accuracy());
	}

	public static void main(String[] args) throws IOException {

		// String zipName = "2018-07-07-22-21-28smallTests.zip";

		String zipName = "2018-07-08-20-18-09little.zip";
		String fileName = DomainUtils.getDropboxLocal() + zipName;
		// test(fileName);

		zipName = "2018-07-08-12-13-18smallTests.zip";
		fileName = DomainUtils.getDropboxLocal() + zipName;
		test(fileName);

	}

}
