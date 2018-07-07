package client;

import java.io.IOException;

import org.deeplearning4j.eval.Evaluation;

import faultrecordreader.ReducedFaultRecordReader;

public class TestMegaTrain {

	public static void main(String[] args) throws IOException {
		String fileName = "models/simpleCNN_megaTrain.zip";
		FaultClassifier classifier = new FaultClassifier(fileName);
		Evaluation evaluation = classifier.evaluate(1, 10000, new ReducedFaultRecordReader());
		System.out.println(evaluation.stats());

	}

}
